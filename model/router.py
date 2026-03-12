"""
model/router.py
───────────────
MoR routing modules — Expert-choice and Token-choice routers.

Key improvements vs the reference repo:
  ① Proper training signal via BCE auxiliary loss
  ② Z-loss regularisation to prevent router logit collapse
  ③ Cosine capacity warmup (model starts with full depth, learns to route)
  ④ Clean, unified API for both routing types
  ⑤ Aux router option (separate detached head for BCE signal)

Theory:
  Expert-choice: the *model* decides which top-k tokens get deep processing.
    → Balanced computation, no variable-length complexity in expert-choice mode.
    → Router: g = σ(θᵀ · h), top-k on g per batch.

  Token-choice: each *token* chooses whether to continue recursing.
    → Variable-length sequences; handled with padding.
    → More faithful to "early exit" intuition.
"""
from __future__ import annotations
from typing import NamedTuple, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
#  Output containers
# ─────────────────────────────────────────────────────────────────

class RouterOutput(NamedTuple):
    """All information produced by a router forward pass."""
    selected_tokens: torch.Tensor     # (B, k, 1) — indices of routed tokens
    gate_weights:    torch.Tensor     # (B, k, 1) — gating values at selected tokens
    router_logits:   torch.Tensor     # (B, T, 1) — raw router logits
    aux_loss:        Optional[torch.Tensor] = None   # scalar BCE loss
    z_loss:          Optional[torch.Tensor] = None   # scalar z-loss


# ─────────────────────────────────────────────────────────────────
#  Linear Router Head (shared by both ExpertChoice and TokenChoice)
# ─────────────────────────────────────────────────────────────────

class LinearRouter(nn.Module):
    """
    Lightweight linear projection that maps each token's hidden state
    to a scalar routing score.

        logit(x) = W · x         (no bias, no activation here)
        score    = σ(logit / T)  (applied externally)

    Args:
        d_model: hidden dimension
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.linear.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d) → logits: (B, T, 1)"""
        return self.linear(x)


# ─────────────────────────────────────────────────────────────────
#  Expert-Choice Router
# ─────────────────────────────────────────────────────────────────

class ExpertChoiceRouter(nn.Module):
    """
    Expert-choice routing (model selects which tokens get deep processing).

    Training:
      The main router is trained end-to-end. Two auxiliary objectives:
        • BCE aux_loss: supervise the router to predict its own top-k mask
        • z-loss: penalise large router logits (router stability)

    Capacity warmup:
      At step 0, capacity_factor = 1.0 (all tokens pass through).
      Over `cap_warmup_steps` steps it decays via cosine to `target_capacity`.
      This prevents cold-start routing collapse.

    Args:
        d_model:          hidden dimension
        target_capacity:  fraction of tokens to process (e.g. 0.5 = 50%)
        router_func:      activation for gate weights ('sigmoid' | 'tanh')
        alpha:            scale applied to gate weights
        sampling:         aux loss type ('aux_loss' | 'aux_router' | 'none')
        aux_loss_weight:  weight for BCE aux loss
        z_loss_weight:    weight for z-loss regularisation
        cap_warmup_steps: steps over which capacity anneals from 1.0 to target
        temp:             temperature scaling on router logits
        rand_router:      if True, use random scores (ablation / debugging)
    """
    def __init__(
        self,
        d_model:          int,
        target_capacity:  float = 1.0,
        router_func:      str   = "sigmoid",
        alpha:            float = 0.1,
        sampling:         str   = "aux_loss",
        aux_loss_weight:  float = 0.001,
        z_loss_weight:    float = 1e-3,
        cap_warmup_steps: int   = 1000,
        temp:             float = 1.0,
        rand_router:      bool  = False,
    ):
        super().__init__()
        self.target_capacity  = target_capacity
        self.router_func      = router_func
        self.alpha            = alpha
        self.sampling         = sampling
        self.aux_loss_weight  = aux_loss_weight
        self.z_loss_weight    = z_loss_weight
        self.cap_warmup_steps = cap_warmup_steps
        self.temp             = temp
        self.rand_router      = rand_router

        self._step  = 0
        self.bce    = nn.BCEWithLogitsLoss(reduction="sum")

        if not rand_router:
            self.router = LinearRouter(d_model)
        if sampling == "aux_router":
            # Separate detached head for generating the BCE target signal
            self.aux_router = LinearRouter(d_model)

    @property
    def current_capacity(self) -> float:
        """Compute current effective capacity (with cosine warmup)."""
        if self.cap_warmup_steps <= 0 or not self.training:
            return self.target_capacity
        ratio  = min(1.0, self._step / self.cap_warmup_steps)
        decay  = 0.5 * (1.0 + math.cos(math.pi * ratio))   # 1.0 → 0.0
        return self.target_capacity + (1.0 - self.target_capacity) * decay

    def forward(
        self,
        x: torch.Tensor,
        prev_selected: Optional[torch.Tensor] = None,
    ) -> RouterOutput:
        """
        Args:
            x:             (B, T, d_model) hidden states (only active tokens)
            prev_selected: (B, k_prev, 1) — indices of tokens active in previous
                           recursion step (for hierarchical filtering). If None,
                           all tokens are considered.

        Returns:
            RouterOutput with selected_tokens, gate_weights, etc.
        """
        if self.training:
            self._step += 1

        B, T, d = x.shape
        cap = self.current_capacity
        top_k = max(1, int(cap * T))

        # ── Get router logits ──
        if self.rand_router:
            raw_logits = torch.rand(B, T, 1, device=x.device, dtype=x.dtype)
        else:
            raw_logits = self.router(x / self.temp)          # (B, T, 1)

        # ── Apply activation to get gating values ──
        if self.router_func == "sigmoid":
            gate_values = torch.sigmoid(raw_logits) * self.alpha
        elif self.router_func == "tanh":
            gate_values = torch.tanh(raw_logits) * self.alpha
        else:
            gate_values = raw_logits

        # ── Select top-k tokens by gate value ──
        sel_vals, selected = torch.topk(gate_values, top_k, dim=1, sorted=False)   # (B, k, 1)

        # Sort indices to maintain causal (positional) order — IMPORTANT for attention
        selected, sort_order = torch.sort(selected, dim=1)
        gate_weights = torch.gather(sel_vals, dim=1, index=sort_order)

        # ── Map local indices back to full-sequence indices (hierarchical) ──
        if prev_selected is not None:
            selected = torch.gather(prev_selected, dim=1, index=selected)

        # ── Compute auxiliary losses (training only) ──
        aux_loss = None
        z_loss   = None

        if self.training and not self.rand_router:
            # Build targets: 1 for selected tokens, 0 for the rest
            targets = torch.zeros(B, T, 1, device=x.device, dtype=gate_values.dtype)
            local_sel = torch.topk(gate_values, top_k, dim=1, sorted=False)[1]     # (B, k, 1)
            targets.scatter_(1, local_sel, 1.0)

            if self.sampling == "aux_loss":
                # BCE on raw router logits vs top-k binary mask
                loss_raw = self.bce(raw_logits.view(-1), targets.view(-1))
                aux_loss = self.aux_loss_weight * loss_raw / (B * T)

            elif self.sampling == "aux_router":
                # Separate detached router head provides target signal
                aux_logits = self.aux_router(x.detach())
                loss_raw   = self.bce(aux_logits.view(-1), targets.view(-1))
                aux_loss   = self.aux_loss_weight * loss_raw / (B * T)

            # Z-loss: penalise large logits (Zoph et al., 2022)
            if self.z_loss_weight > 0:
                z = torch.logsumexp(raw_logits.squeeze(-1), dim=-1)    # (B,)
                z_loss = self.z_loss_weight * z.pow(2).mean()

        return RouterOutput(
            selected_tokens=selected,
            gate_weights=gate_weights,
            router_logits=raw_logits,
            aux_loss=aux_loss,
            z_loss=z_loss,
        )


# ─────────────────────────────────────────────────────────────────
#  Token-Choice Router
# ─────────────────────────────────────────────────────────────────

class TokenChoiceRouter(nn.Module):
    """
    Token-choice routing — each token decides whether to recurse deeper.

    Unlike expert-choice, token-choice can produce variable-length active sets.
    We handle this with padding so that batched computation remains efficient.

    The routing decision is a hard threshold on σ(logit) ≥ 0.5,
    but during training we use soft gates to keep gradients flowing.

    Args:
        d_model:         hidden dimension
        threshold:       sigmoid threshold for binary gating
        balancing_loss_weight: load-balancing loss weight
        z_loss_weight:   z-loss weight
    """
    def __init__(
        self,
        d_model:                int,
        threshold:              float = 0.5,
        balancing_loss_weight:  float = 0.01,
        z_loss_weight:          float = 1e-3,
        temp:                   float = 1.0,
    ):
        super().__init__()
        self.threshold             = threshold
        self.balancing_loss_weight = balancing_loss_weight
        self.z_loss_weight         = z_loss_weight
        self.temp                  = temp

        self.router = LinearRouter(d_model)

    def forward(
        self,
        x: torch.Tensor,
        prev_selected: Optional[torch.Tensor] = None,
    ) -> RouterOutput:
        """
        Args:
            x:             (B, T, d_model) — all tokens (or currently active)
            prev_selected: (B, k_prev, 1)  — active indices from previous step

        Returns:
            RouterOutput
        """
        B, T, d = x.shape

        raw_logits  = self.router(x / self.temp)           # (B, T, 1)
        gate_values = torch.sigmoid(raw_logits)             # (B, T, 1)

        # Hard selection: tokens where gate > threshold
        selected_mask = (gate_values.squeeze(-1) >= self.threshold)  # (B, T)

        # Convert mask to indices — pad to max active length
        selected_list = []
        gate_list     = []
        max_k = max(1, selected_mask.sum(dim=-1).max().item())

        for b in range(B):
            idx = selected_mask[b].nonzero(as_tuple=False).view(-1, 1)   # (k_b, 1)
            gv  = gate_values[b][selected_mask[b]]                        # (k_b,)
            pad = max_k - idx.shape[0]
            if pad > 0:
                idx = torch.cat([idx, idx[-1:].expand(pad, 1)], dim=0)
                gv  = torch.cat([gv, gv[-1:].expand(pad)], dim=0)
            selected_list.append(idx)
            gate_list.append(gv)

        selected     = torch.stack(selected_list, dim=0).unsqueeze(-1)    # (B, max_k, 1)
        gate_weights = torch.stack(gate_list, dim=0).unsqueeze(-1)        # (B, max_k, 1)

        # Map to global indices via prev_selected
        if prev_selected is not None:
            selected = torch.gather(prev_selected, dim=1, index=selected)

        # ── Losses ──
        aux_loss = None
        z_loss   = None

        if self.training:
            # Load-balancing: encourage uniform token distribution across steps
            mean_gate = gate_values.mean()
            aux_loss  = self.balancing_loss_weight * mean_gate * (1 - mean_gate)

            if self.z_loss_weight > 0:
                z = torch.logsumexp(raw_logits.squeeze(-1), dim=-1)
                z_loss = self.z_loss_weight * z.pow(2).mean()

        return RouterOutput(
            selected_tokens=selected,
            gate_weights=gate_weights,
            router_logits=raw_logits,
            aux_loss=aux_loss,
            z_loss=z_loss,
        )


# ─────────────────────────────────────────────────────────────────
#  Router Registry
# ─────────────────────────────────────────────────────────────────

ROUTER_TYPES = {
    "expert": ExpertChoiceRouter,
    "token":  TokenChoiceRouter,
}
