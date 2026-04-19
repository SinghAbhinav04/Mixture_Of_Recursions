"""
model/base.py
─────────────
LLaMA-style causal transformer backbone.

Architecture choices (same as Llama 2/3):
  • RMSNorm         instead of LayerNorm
  • RoPE            instead of learned positional embeddings
  • SwiGLU          instead of standard GELU FFN
  • Pre-norm        (norm before attention/FFN, not after)
  • GQA-ready       (n_kv_heads ≤ n_heads via repeat_kv)
  • No bias         in linear layers (following modern LLMs)

This module provides:
  - RMSNorm
  - RotaryEmbedding
  - CausalSelfAttention  (with GQA support)
  - SwiGLUFFN
  - TransformerBlock     (the single reusable block for MoR sharing)
  - CausalTransformer    (full stacked model, used as baseline / base class)
"""
from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
#  RMSNorm
# ─────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019).
    Cheaper than LayerNorm — no mean subtraction.
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


# ─────────────────────────────────────────────────────────────────
#  Rotary Positional Embedding (RoPE)
# ─────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    Rotary positional embeddings (Su et al., 2021 — RoFormer).
    Applied per-head to Q and K before dot-product.
    """
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10_000.0):
        super().__init__()
        # Inverse frequencies: θ_i = 1 / base^(2i/head_dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute cos/sin cache for efficiency
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # (seq_len, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)        # (seq_len, head_dim)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) sliced to the sequence length of x."""
        seq_len = x.shape[2] + offset
        if seq_len > self.cos_cache.shape[0]:
            self._build_cache(seq_len)
        cos = self.cos_cache[offset:offset + x.shape[2]]
        sin = self.sin_cache[offset:offset + x.shape[2]]
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the latter half of x into the former half (RoPE rotation)."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.
    q, k : (B, n_heads, T, head_dim)
    cos, sin : (T, head_dim) — broadcast across B and heads
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


# ─────────────────────────────────────────────────────────────────
#  GQA Utility
# ─────────────────────────────────────────────────────────────────

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat K/V heads to match the number of Q heads (GQA → MHA expansion).
    x : (B, n_kv_heads, T, head_dim)
    returns : (B, n_heads, T, head_dim)
    """
    if n_rep == 1:
        return x
    B, n_kv, T, hd = x.shape
    return x[:, :, None, :, :].expand(B, n_kv, n_rep, T, hd).reshape(B, n_kv * n_rep, T, hd)


# ─────────────────────────────────────────────────────────────────
#  Causal Self-Attention
# ─────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with optional GQA.

    When n_kv_heads < n_heads, this implements Grouped-Query Attention (GQA):
    multiple Q heads share the same K/V head.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        rope_base: float = 10_000.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads     = n_heads
        self.n_kv_heads  = n_kv_heads
        self.n_rep       = n_heads // n_kv_heads          # repetitions for GQA
        self.head_dim    = d_model // n_heads
        self.d_model     = d_model

        # Projections — no bias (LLaMA style)
        self.q_proj  = nn.Linear(d_model, n_heads    * self.head_dim, bias=False)
        self.k_proj  = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj  = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj  = nn.Linear(d_model, d_model,                    bias=False)

        self.dropout   = dropout
        self.attn_drop = nn.Dropout(dropout)

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len, rope_base)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape
        offset = past_kv[0].shape[2] if past_kv is not None else 0

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary(q, offset=offset)
        q, k = apply_rotary_emb(q, k, cos, sin)

        # KV cache
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v) if use_cache else None

        # GQA: expand K/V to match Q heads
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Use PyTorch's optimised SDPA (Flash Attention when available)
        # For prefill (no cache, T > 1): is_causal=True handles the mask internally
        # For decode  (with cache, T = 1): no mask needed (single query attends to all past)
        if past_kv is None and T > 1:
            # Prefill: pure causal — SDPA builds the mask internally (O(1) memory)
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Decode (T=1) or cached continuation: no causal mask needed
            # because a single new token can attend to all cached keys
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )

        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o_proj(out), new_kv


# ─────────────────────────────────────────────────────────────────
#  SwiGLU Feed-Forward Network
# ─────────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward network (used in LLaMA, PaLM, Gemma, etc.).

    FFN(x) = SiLU(W_gate · x) ⊙ (W_up · x)  → W_down
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff,   d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))


# ─────────────────────────────────────────────────────────────────
#  Transformer Block
# ─────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Single decoder block: pre-norm attention + pre-norm FFN.
    This is the unit that gets SHARED across virtual layers in MoR.

    Pre-norm formulation:
        x = x + Attention(LN(x))
        x = x + FFN(LN(x))
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        rope_base: float = 10_000.0,
    ):
        super().__init__()
        self.attn     = CausalSelfAttention(d_model, n_heads, n_kv_heads, max_seq_len, dropout, rope_base)
        self.ffn      = SwiGLUFFN(d_model, d_ff, dropout)
        self.ln_attn  = RMSNorm(d_model, norm_eps)
        self.ln_ffn   = RMSNorm(d_model, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        # Pre-norm self-attention residual
        attn_out, new_kv = self.attn(self.ln_attn(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        # Pre-norm FFN residual
        x = x + self.ffn(self.ln_ffn(x))
        return x, new_kv


# ─────────────────────────────────────────────────────────────────
#  Full Causal Transformer (Baseline — no MoR)
# ─────────────────────────────────────────────────────────────────

class CausalTransformer(nn.Module):
    """
    Standard stacked causal transformer (no MoR).
    Used as the baseline and as the base class for MoRForCausalLM.

    Architecture: [Embed] → [Block × n_layers] → [RMSNorm] → [LM Head]
    """
    def __init__(self, cfg):
        """
        Args:
            cfg: ModelConfig dataclass instance
        """
        super().__init__()
        self.cfg = cfg

        self.tok_emb  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model     = cfg.d_model,
                n_heads     = cfg.n_heads,
                n_kv_heads  = cfg.n_kv_heads,
                d_ff        = cfg.d_ff,
                max_seq_len = cfg.max_seq_len,
                dropout     = cfg.dropout,
                norm_eps    = cfg.norm_eps,
                rope_base   = cfg.rope_base,
            )
            for _ in range(cfg.n_layers)
        ])

        self.norm    = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: embed and lm_head share weights (common LLM trick)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.cfg.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=self.cfg.init_std)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        past_kvs: Optional[list] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            idx:      (B, T) token indices
            targets:  (B, T) target indices for loss (optional)
            past_kvs: list of (k, v) tuples per layer for KV cache
            use_cache: whether to return KV cache for next step

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss (or None)
        """
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"

        x = self.emb_drop(self.tok_emb(idx))   # (B, T, d_model)

        new_kvs = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            pkv = past_kvs[i] if past_kvs is not None else None
            x, new_kv = layer(x, past_kv=pkv, use_cache=use_cache)
            if use_cache:
                new_kvs.append(new_kv)

        x = self.norm(x)
        logits = self.lm_head(x)                # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        if use_cache:
            return logits, loss, new_kvs
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with temperature, top-k, and top-p sampling.

        Args:
            prompt_ids:     (1, T_prompt) or (B, T_prompt) token indices
            max_new_tokens: number of tokens to generate
            temperature:    sampling temperature (1.0 = no scaling)
            top_k:          keep only top-k logits before sampling
            top_p:          nucleus sampling threshold (None = disable)

        Returns:
            (B, T_prompt + max_new_tokens) token indices
        """
        self.eval()
        idx = prompt_ids

        for _ in range(max_new_tokens):
            # Crop to last max_seq_len tokens
            idx_cond = idx[:, -self.cfg.max_seq_len:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]           # (B, vocab_size)

            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens beyond the nucleus
                sorted_logits[cumprobs - F.softmax(sorted_logits, dim=-1) > top_p] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, idx_next], dim=1)

        return idx

    def num_params(self) -> dict:
        total  = sum(p.numel() for p in self.parameters())
        unique = sum(p.numel() for p in set(self.parameters()))
        return {"total": total, "unique": unique}
