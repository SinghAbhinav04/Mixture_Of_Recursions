"""
model/mor_layer.py
──────────────────
MoR Decoder Layer — the heart of the Mixture-of-Recursions architecture.

The critical improvement over a naive MoR (like gpt_mor.py) is that we
PHYSICALLY REDUCE the sequence length before passing tokens to the block:

    gather(x, top_k_indices)           # [B, T, D] → [B, k, D]  ← REAL speedup
    out = transformer_block(reduced)   # only k tokens processed
    scatter_add(total_x, indices, out) # merge gated result back

This is where the actual compute savings happen. By selecting the top-k
tokens, we process a subsequence of length k < T through the expensive
attention and FFN, giving us sub-quadratic effective computation.

Two variants are provided:
  • ExpertChoiceMoRLayer  — model selects top-k tokens (recommended)
  • TokenChoiceMoRLayer   — tokens self-select via threshold
"""
from __future__ import annotations
from typing import List, NamedTuple, Optional, Tuple
import torch
import torch.nn as nn

from model.base import TransformerBlock
from model.router import ExpertChoiceRouter, TokenChoiceRouter, RouterOutput


# ─────────────────────────────────────────────────────────────────
#  Layer Output Container
# ─────────────────────────────────────────────────────────────────

class MoRLayerOutput(NamedTuple):
    """Everything returned by a MoR layer forward pass."""
    hidden_states:    torch.Tensor              # (B, T, d)
    selected_tokens:  Optional[torch.Tensor]   # (B, k, 1) last-recursion selection
    aux_loss:         Optional[torch.Tensor]   # scalar
    z_loss:           Optional[torch.Tensor]   # scalar
    router_logits:    Optional[torch.Tensor]   # (B, T, 1)


# ─────────────────────────────────────────────────────────────────
#  Expert-Choice MoR Layer
# ─────────────────────────────────────────────────────────────────

class ExpertChoiceMoRLayer(nn.Module):
    """
    MoR decoder layer using expert-choice routing.

    One ExpertChoiceMoRLayer wraps a group of shared TransformerBlocks
    and a router. It implements one recursion step:

      1. Router scores all active tokens.
      2. Top-k tokens (by score) are selected → gathered into (B, k, D).
      3. The gathered subsequence is passed through each shared block.
      4. Results are scatter-added back into the full hidden state,
         weighted by the router gate value.
      5. Return the updated hidden state and metadata (aux_loss, z_loss, etc.)

    Args:
        blocks:           list of shared TransformerBlock modules
        router:           ExpertChoiceRouter for this recursion step
        gating:           'weighted' (multiply by gate) or 'identity' (no scaling)
    """
    def __init__(
        self,
        blocks: nn.ModuleList,
        router: ExpertChoiceRouter,
        gating: str = "weighted",
    ):
        super().__init__()
        self.blocks  = blocks
        self.router  = router
        self.gating  = gating
        self.is_mor  = True
        self.mor_type = "expert"

    def forward(
        self,
        x:              torch.Tensor,              # (B, T, d_model) — all positions
        prev_selected:  Optional[torch.Tensor],    # (B, k_prev, 1) from last recursion
        **kwargs,
    ) -> MoRLayerOutput:
        B, T, d = x.shape
        total_x = x                                # keep full sequence for scatter

        # ── 1. Get only the currently-active tokens ──
        if prev_selected is not None:
            # restrict computation to positions active in the previous step
            active_x = torch.gather(x, 1, prev_selected.expand(-1, -1, d))  # (B, k_prev, d)
        else:
            active_x = x                           # (B, T, d) — first recursion: all tokens

        # ── 2. Expert-choice routing ──
        router_out: RouterOutput = self.router(active_x, prev_selected=prev_selected)
        selected   = router_out.selected_tokens    # (B, k, 1) — global indices
        gate_w     = router_out.gate_weights       # (B, k, 1)

        # ── 3. Gather selected tokens ──
        k = selected.shape[1]
        idx_exp = selected.expand(-1, -1, d)       # (B, k, d)
        top_k_x = torch.gather(x, 1, idx_exp)      # (B, k, d) — PHYSICAL reduction

        # ── 4. Process through shared blocks ──
        h = top_k_x
        for block in self.blocks:
            h, _ = block(h)                        # (B, k, d)

        # ── 5. Apply gating ──
        if self.gating == "weighted":
            h_out = h * gate_w                     # (B, k, d) — element-wise scale
        else:
            h_out = h                              # identity gate

        # ── 6. Scatter-add back into full hidden state ──
        total_x = total_x.scatter_add(1, idx_exp, h_out)   # (B, T, d)

        return MoRLayerOutput(
            hidden_states   = total_x,
            selected_tokens = selected,
            aux_loss        = router_out.aux_loss,
            z_loss          = router_out.z_loss,
            router_logits   = router_out.router_logits,
        )


# ─────────────────────────────────────────────────────────────────
#  Token-Choice MoR Layer
# ─────────────────────────────────────────────────────────────────

class TokenChoiceMoRLayer(nn.Module):
    """
    MoR decoder layer using token-choice routing.

    Token-choice lets each token decide whether to recurse deeper by
    thresholding a per-token sigmoid score. Variable-length active sets
    are handled via padding (from the router).

    Args:
        blocks:  shared TransformerBlock modules
        router:  TokenChoiceRouter
        gating:  'weighted' | 'identity'
    """
    def __init__(
        self,
        blocks: nn.ModuleList,
        router: TokenChoiceRouter,
        gating: str = "weighted",
    ):
        super().__init__()
        self.blocks   = blocks
        self.router   = router
        self.gating   = gating
        self.is_mor   = True
        self.mor_type = "token"

    def forward(
        self,
        x:             torch.Tensor,
        prev_selected: Optional[torch.Tensor],
        **kwargs,
    ) -> MoRLayerOutput:
        B, T, d = x.shape
        total_x = x

        if prev_selected is not None:
            active_x = torch.gather(x, 1, prev_selected.expand(-1, -1, d))
        else:
            active_x = x

        # ── Routing ──
        router_out: RouterOutput = self.router(active_x, prev_selected=prev_selected)
        selected = router_out.selected_tokens    # (B, k, 1)
        gate_w   = router_out.gate_weights       # (B, k, 1)

        # ── Gather ──
        idx_exp = selected.expand(-1, -1, d)
        top_k_x = torch.gather(x, 1, idx_exp)

        # ── Compute ──
        h = top_k_x
        for block in self.blocks:
            h, _ = block(h)

        # ── Gate and scatter ──
        h_out   = h * gate_w if self.gating == "weighted" else h
        total_x = total_x.scatter_add(1, idx_exp, h_out)

        return MoRLayerOutput(
            hidden_states   = total_x,
            selected_tokens = selected,
            aux_loss        = router_out.aux_loss,
            z_loss          = router_out.z_loss,
            router_logits   = router_out.router_logits,
        )


# ─────────────────────────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────────────────────────

MOR_LAYER_TYPES = {
    "expert": ExpertChoiceMoRLayer,
    "token":  TokenChoiceMoRLayer,
}
