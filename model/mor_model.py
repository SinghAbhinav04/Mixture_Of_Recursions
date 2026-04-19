"""
model/mor_model.py
──────────────────
MoRForCausalLM — the full Mixture-of-Recursions language model.

This module ties everything together:
  • A LLaMA-style backbone (from model/base.py)
  • Parameter-sharing via sharing strategies (model/sharing.py)
  • MoR routing layers (model/mor_layer.py)
  • Recursion-wise KV caching (model/kv_cache.py)

Model assembly flow:
  1. Instantiate CausalTransformer (full n_layers of TransformerBlocks)
  2. Call `transform_to_mor()` to:
       a. Count unique blocks needed for the chosen strategy
       b. Merge / alias blocks to implement parameter sharing
       c. Wrap the model with MoR routing layers (one per recursion step)
  3. Forward pass runs the MoR loop:
       For each recursion r:
           → router selects top-k tokens (by capacity)
           → gather(x, top_k_indices) → run shared block group → scatter back
       → final norm → lm_head → logits, loss

Losses collected during training:
  • cross-entropy LM loss  (primary)
  • aux BCE loss           (router supervision)
  • z-loss                 (router stability)
  All three are summed into a single scalar returned as `loss`.
"""
from __future__ import annotations
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base import TransformerBlock, RMSNorm, CausalTransformer
from model.sharing import build_routing_table, count_unique_blocks, print_routing_table
from model.router import ExpertChoiceRouter, TokenChoiceRouter
from model.mor_layer import ExpertChoiceMoRLayer, TokenChoiceMoRLayer, MoRLayerOutput
from model.kv_cache import RecursiveKVCache
from configs.default import ModelConfig, MoRConfig


# ─────────────────────────────────────────────────────────────────
#  Model Output Container
# ─────────────────────────────────────────────────────────────────

class MoRModelOutput(NamedTuple):
    logits:        torch.Tensor              # (B, T, vocab_size)
    loss:          Optional[torch.Tensor]    # total loss scalar
    lm_loss:       Optional[torch.Tensor]    # cross-entropy component
    aux_loss:      Optional[torch.Tensor]    # router BCE component
    z_loss:        Optional[torch.Tensor]    # router z-loss component
    router_logits: Optional[List[torch.Tensor]]  # per-recursion router logits


# ─────────────────────────────────────────────────────────────────
#  MoR For Causal LM
# ─────────────────────────────────────────────────────────────────

class MoRForCausalLM(nn.Module):
    """
    Full Mixture-of-Recursions causal language model.

    Usage:
        model_cfg = ModelConfig(d_model=512, n_layers=12, n_heads=8)
        mor_cfg   = MoRConfig(n_recursions=3, strategy="middle_cycle")
        model     = MoRForCausalLM(model_cfg, mor_cfg)

        logits, loss, *_ = model(input_ids, targets)

    Args:
        model_cfg: ModelConfig — architecture hyperparameters
        mor_cfg:   MoRConfig  — routing / sharing hyperparameters
    """
    def __init__(self, model_cfg: ModelConfig, mor_cfg: MoRConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.mor_cfg   = mor_cfg

        # ── 1. Build the unique shared TransformerBlocks ──
        n_unique = count_unique_blocks(mor_cfg.strategy, model_cfg.n_layers, mor_cfg.n_recursions)
        self.shared_blocks = nn.ModuleList([
            TransformerBlock(
                d_model     = model_cfg.d_model,
                n_heads     = model_cfg.n_heads,
                n_kv_heads  = model_cfg.n_kv_heads,
                d_ff        = model_cfg.d_ff,
                max_seq_len = model_cfg.max_seq_len,
                dropout     = model_cfg.dropout,
                norm_eps    = model_cfg.norm_eps,
                rope_base   = model_cfg.rope_base,
            )
            for _ in range(n_unique)
        ])

        # ── 2. Build routing table: virtual_layer_ℓ → unique_block_idx ──
        self.routing_table = build_routing_table(
            mor_cfg.strategy, model_cfg.n_layers, mor_cfg.n_recursions
        )

        # ── 3. Compute capacity per recursion step ──
        N = mor_cfg.n_recursions
        if mor_cfg.router.capacity is not None:
            capacities = mor_cfg.router.capacity
        else:
            # Paper default: step j processes (N-j)/N fraction of tokens
            capacities = [(N - j) / N for j in range(N)]
        assert len(capacities) == N, \
            f"Expected {N} capacity values, got {len(capacities)}"

        # ── 4. Build MoR layers (one per recursion step) ──
        # Each MoR layer wraps the block group for that recursion step
        layers_per_step = model_cfg.n_layers // N
        router_cfg = mor_cfg.router

        self.mor_layers = nn.ModuleList()
        for r in range(N):
            # Get the block index range for this recursion step
            step_start = r * layers_per_step
            step_end   = step_start + layers_per_step
            block_indices = [self.routing_table[ell] for ell in range(step_start, min(step_end, model_cfg.n_layers))]
            step_blocks = nn.ModuleList([self.shared_blocks[i] for i in block_indices])

            if mor_cfg.routing_type == "expert":
                router = ExpertChoiceRouter(
                    d_model          = model_cfg.d_model,
                    target_capacity  = capacities[r],
                    router_func      = router_cfg.router_func,
                    alpha            = router_cfg.alpha,
                    sampling         = router_cfg.sampling,
                    aux_loss_weight  = router_cfg.aux_loss_weight,
                    z_loss_weight    = router_cfg.z_loss_weight,
                    cap_warmup_steps = router_cfg.cap_warmup_steps,
                    temp             = router_cfg.temp,
                    rand_router      = router_cfg.rand_router,
                    router_type      = router_cfg.router_type,
                )
                layer = ExpertChoiceMoRLayer(step_blocks, router, gating=mor_cfg.gating, block_indices=block_indices)
            else:
                router = TokenChoiceRouter(
                    d_model               = model_cfg.d_model,
                    n_recursions          = N,
                    balancing_loss_weight = router_cfg.aux_loss_weight,
                    z_loss_weight         = router_cfg.z_loss_weight,
                    temp                  = router_cfg.temp,
                    router_type           = router_cfg.router_type,
                    balancing             = router_cfg.balancing,
                    bal_warmup_steps      = router_cfg.bal_warmup_steps,
                )
                layer = TokenChoiceMoRLayer(step_blocks, router, gating=mor_cfg.gating, block_indices=block_indices)

            self.mor_layers.append(layer)

        # ── 5. Embeddings, norm, and output head ──
        self.tok_emb   = nn.Embedding(model_cfg.vocab_size, model_cfg.d_model)
        self.emb_drop  = nn.Dropout(model_cfg.dropout)
        self.norm      = RMSNorm(model_cfg.d_model, model_cfg.norm_eps)
        self.lm_head   = nn.Linear(model_cfg.d_model, model_cfg.vocab_size, bias=False)

        # Weight tying
        if model_cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

        # ── Summary ──
        print_routing_table(mor_cfg.strategy, model_cfg.n_layers, mor_cfg.n_recursions)
        print(f"[MoRForCausalLM] capacities={[round(c, 3) for c in capacities]}")
        p = self.num_params()
        print(f"[MoRForCausalLM] {p['total']/1e6:.2f}M total params | "
              f"{p['unique']/1e6:.2f}M unique (non-shared) params")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.model_cfg.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=self.model_cfg.init_std)

    def forward(
        self,
        idx:       torch.Tensor,                   # (B, T)
        targets:   Optional[torch.Tensor] = None,  # (B, T)
        use_cache: bool = False,
        kv_cache:  Optional['RecursiveKVCache'] = None,
    ) -> MoRModelOutput:
        """
        Forward pass through the MoR model.

        Args:
            idx:     (B, T) LongTensor of token indices
            targets: (B, T) LongTensor of next-token targets (for loss)

        Returns:
            MoRModelOutput with logits, total_loss, and loss components
        """
        B, T = idx.shape
        assert T <= self.model_cfg.max_seq_len, \
            f"Input length {T} exceeds max_seq_len {self.model_cfg.max_seq_len}"

        # ── Token embedding ──
        x = self.emb_drop(self.tok_emb(idx))           # (B, T, d_model)

        # ── Reset token-choice router caches (assignments are per-sequence) ──
        for mor_layer in self.mor_layers:
            if hasattr(mor_layer.router, 'reset'):
                mor_layer.router.reset()

        # ── MoR recursion loop ──
        prev_selected    = None
        total_aux_loss   = torch.tensor(0.0, device=idx.device)
        total_z_loss     = torch.tensor(0.0, device=idx.device)
        all_router_logits: List[torch.Tensor] = []

        for r, mor_layer in enumerate(self.mor_layers):
            layer_out: MoRLayerOutput = mor_layer(
                x, 
                prev_selected=prev_selected,
                recursion_idx=r,
                kv_cache=kv_cache,
                use_cache=use_cache
            )

            x             = layer_out.hidden_states
            prev_selected = layer_out.selected_tokens

            if layer_out.aux_loss is not None:
                total_aux_loss = total_aux_loss + layer_out.aux_loss
            if layer_out.z_loss is not None:
                total_z_loss = total_z_loss + layer_out.z_loss
            if layer_out.router_logits is not None:
                all_router_logits.append(layer_out.router_logits)

        # ── Output head ──
        x      = self.norm(x)
        logits = self.lm_head(x)                        # (B, T, vocab_size)

        # ── Losses ──
        lm_loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        loss = None
        if lm_loss is not None:
            loss = lm_loss + total_aux_loss + total_z_loss

        return MoRModelOutput(
            logits        = logits,
            loss          = loss,
            lm_loss       = lm_loss,
            aux_loss      = total_aux_loss if targets is not None else None,
            z_loss        = total_z_loss   if targets is not None else None,
            router_logits = all_router_logits if all_router_logits else None,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids:     torch.Tensor,
        max_new_tokens: int = 200,
        temperature:    float = 0.8,
        top_k:          Optional[int] = 50,
        top_p:          Optional[float] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.

        During inference all tokens pass through all MoR layers (full capacity).
        The router still scores and weights tokens, but nothing is truly dropped
        since we need coherent generations across the full context.

        Args:
            prompt_ids:     (1, T) or (B, T) LongTensor
            max_new_tokens: tokens to generate
            temperature:    sampling temperature (1.0 = no change)
            top_k:          top-k filtering
            top_p:          nucleus filtering

        Returns:
            (B, T + max_new_tokens) token tensor
        """
        self.eval()
        idx = prompt_ids

        # ── 1. Init KV Cache ──
        kv_cache = RecursiveKVCache(
            n_unique_blocks=len(self.shared_blocks),
            n_recursions=self.mor_cfg.n_recursions,
            n_kv_heads=self.model_cfg.n_kv_heads,
            head_dim=self.model_cfg.d_model // self.model_cfg.n_heads,
        )

        # ── 2. Prefill Phase (Full prompt through network) ──
        idx_cond = idx[:, -self.model_cfg.max_seq_len:]
        out = self(idx_cond, use_cache=True, kv_cache=kv_cache)
        logits = out.logits[:, -1, :]

        # ── 3. Generation Loop (only pass new tokens) ──
        for _ in range(max_new_tokens):

            if temperature != 1.0:
                logits = logits / temperature

            # Top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus)
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_logits[cum_probs - F.softmax(sorted_logits, dim=-1) > top_p] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs    = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, idx_next], dim=1)

            # Optimisation: Next step only passes the newest token
            out = self(idx_next, use_cache=True, kv_cache=kv_cache)
            logits = out.logits[:, -1, :]

        return idx

    def num_params(self) -> Dict[str, int]:
        """
        Count total vs unique (non-shared) parameters.
        PyTorch's .parameters() deduplicates by default, so we manually sum
        across MoR layers to get the 'total' (virtual) param count.
        """
        unique_params = set(self.parameters())
        unique_count = sum(p.numel() for p in unique_params)

        # Start with non-recursing params (embeddings, head, norm)
        # We find these by taking all params and subtracting the ones in shared_blocks
        shared_params = set()
        for b in self.shared_blocks:
            shared_params.update(b.parameters())

        base_count = sum(p.numel() for p in (unique_params - shared_params))

        # Sum parameters of all blocks used in the virtual path
        # (Total = Base + N_recursions * params_per_recursion)
        total_recursion_count = 0
        for layer in self.mor_layers:
            # Each MoR layer has its own router + a group of shared blocks
            layer_total = sum(p.numel() for p in layer.parameters())
            total_recursion_count += layer_total

        return {
            "total": base_count + total_recursion_count,
            "unique": unique_count
        }

    def print_summary(self):
        """Print a human-readable model summary."""
        p = self.num_params()
        cfg = self.model_cfg
        mor = self.mor_cfg
        n_unique = len(self.shared_blocks)
        print("=" * 60)
        print("  Abhinav-MoR: Mixture-of-Recursions LM")
        print("=" * 60)
        print(f"  Architecture:  LLaMA-style (RoPE + RMSNorm + SwiGLU)")
        print(f"  d_model:       {cfg.d_model}")
        print(f"  n_layers:      {cfg.n_layers} virtual  ({n_unique} unique, shared {mor.n_recursions}×)")
        print(f"  n_heads / kv:  {cfg.n_heads} Q  /  {cfg.n_kv_heads} KV  (GQA ratio {cfg.n_heads // cfg.n_kv_heads}×)")
        print(f"  d_ff:          {cfg.d_ff}  (SwiGLU)")
        print(f"  max_seq_len:   {cfg.max_seq_len}")
        print(f"  vocab_size:    {cfg.vocab_size:,}")
        print("-" * 60)
        print(f"  Strategy:      {mor.strategy}")
        print(f"  Routing:       {mor.routing_type}")
        print(f"  N_recursions:  {mor.n_recursions}")
        print(f"  Routing table: {self.routing_table}")
        print("-" * 60)
        print(f"  Total params:  {p['total']/1e6:.2f}M")
        print(f"  Unique params: {p['unique']/1e6:.2f}M  ({100*p['unique']/p['total']:.1f}% of total)")
        print("=" * 60)
