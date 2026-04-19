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
        out_dim: output dimension (1 for expert-choice, N_r for token-choice)
    """
    def __init__(self, d_model: int, out_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(d_model, out_dim, bias=False)
        nn.init.normal_(self.linear.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d) → logits: (B, T, out_dim)"""
        return self.linear(x)


class MLPRouter(nn.Module):
    """
    Two-layer MLP router (Reference: model/mor_model/util.py).
    Better for token-choice routing per paper §4.2.

    Args:
        d_model: hidden dimension
        out_dim: output dimension
    """
    def __init__(self, d_model: int, out_dim: int = 1):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_model * 2, bias=False),
            nn.GELU(),
            nn.Linear(d_model * 2, out_dim, bias=False),
        )
        for layer in self.linear:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class WideMLPRouter(nn.Module):
    """
    Wide two-layer MLP router (same architecture as MLPRouter but
    semantically distinct for config clarity).

    Args:
        d_model: hidden dimension
        out_dim: output dimension
    """
    def __init__(self, d_model: int, out_dim: int = 1):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_model * 2, bias=False),
            nn.GELU(),
            nn.Linear(d_model * 2, out_dim, bias=False),
        )
        for layer in self.linear:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# Router head registry
ROUTER_HEAD_TYPES = {
    "linear":   LinearRouter,
    "mlp":      MLPRouter,
    "wide_mlp": WideMLPRouter,
}


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
        router_type:      str   = "linear",
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
            RouterHead = ROUTER_HEAD_TYPES.get(router_type, LinearRouter)
            self.router = RouterHead(d_model, out_dim=1)
        if sampling == "aux_router":
            # Separate detached head for generating the BCE target signal
            RouterHead = ROUTER_HEAD_TYPES.get(router_type, LinearRouter)
            self.aux_router = RouterHead(d_model, out_dim=1)

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
#  Token-Choice Router (Paper §2.2.1, Eq 2.2)
# ─────────────────────────────────────────────────────────────────

class TokenChoiceRouter(nn.Module):
    """
    Token-choice routing (Paper Eq 2.2) — one-shot assignment.

    Unlike expert-choice (where each recursion step picks its own top-k),
    token-choice commits each token to a FIXED recursion depth at the start.
    The router runs ONCE after recursion 0, producing per-token scores over
    N_r experts (recursion depths). Each token is assigned via argmax (top-1).

    At recursion step r, only tokens assigned to depth >= r participate.

    Supports two balancing strategies (matching reference):
      - 'loss':      gradient-based balancing loss (Σ P_i * f_i)
      - 'loss_free': learnable router_bias parameter shifts routing
                     decisions without gradient-based loss

    Also supports balancing warmup: during the first `bal_warmup_steps`,
    ALL tokens are forced to the deepest expert, giving the router time
    to learn before making routing decisions.

    Args:
        d_model:      hidden dimension
        n_recursions: number of recursion experts (N_r)
        balancing_loss_weight: load-balancing loss weight
        z_loss_weight: z-loss weight
        temp:         temperature for logit scaling
        router_type:  'linear', 'mlp', or 'wide_mlp'
        balancing:    'loss' or 'loss_free'
        bal_warmup_steps: steps during which all tokens route to deepest
    """
    def __init__(
        self,
        d_model:               int,
        n_recursions:          int   = 3,
        balancing_loss_weight: float = 0.01,
        z_loss_weight:         float = 1e-3,
        temp:                  float = 1.0,
        router_type:           str   = "linear",
        balancing:             str   = "loss",
        bal_warmup_steps:      int   = 0,
    ):
        super().__init__()
        self.n_recursions          = n_recursions
        self.balancing_loss_weight = balancing_loss_weight
        self.z_loss_weight         = z_loss_weight
        self.temp                  = temp
        self.balancing             = balancing
        self.bal_warmup_steps      = bal_warmup_steps
        self._step                 = 0

        # Router head (projects to N_r expert scores)
        RouterHead = ROUTER_HEAD_TYPES.get(router_type, LinearRouter)
        self.router = RouterHead(d_model, out_dim=n_recursions)

        # Loss-free balancing: learnable bias shifts routing decisions
        if balancing == "loss_free":
            self.register_parameter(
                "router_bias",
                nn.Parameter(torch.zeros(n_recursions), requires_grad=False)
            )

        # Cache: after the first call, store assignments for subsequent steps
        self._assignments: Optional[torch.Tensor] = None   # (B, T) ints in [0, N_r)
        self._gate_scores: Optional[torch.Tensor] = None   # (B, T, N_r) softmax
        self._raw_logits:  Optional[torch.Tensor] = None   # (B, T, N_r) raw

    def reset(self):
        """Clear cached assignments (call before each new sequence)."""
        self._assignments = None
        self._gate_scores = None
        self._raw_logits  = None

    def forward(
        self,
        x: torch.Tensor,
        prev_selected: Optional[torch.Tensor] = None,
        recursion_idx: int = 0,
    ) -> RouterOutput:
        """
        Args:
            x:              (B, T, d_model) — hidden states after recursion 0
            prev_selected:  unused for token-choice (kept for API compat)
            recursion_idx:  current recursion step (0-indexed)

        Returns:
            RouterOutput with tokens assigned to depth >= recursion_idx
        """
        B, T, d = x.shape

        if self.training:
            self._step += 1

        # ── First call (recursion_idx == 0): compute assignments ──
        if self._assignments is None or self._assignments.shape[1] != T:
            raw_logits = self.router(x / self.temp)             # (B, T, N_r)
            router_probs = F.softmax(raw_logits, dim=-1)        # (B, T, N_r)

            # Apply loss-free bias if enabled
            if self.balancing == "loss_free" and hasattr(self, "router_bias"):
                router_probs_biased = router_probs + self.router_bias
            else:
                router_probs_biased = router_probs

            # Balancing warmup: force all tokens to deepest expert
            if self.training and self._step < self.bal_warmup_steps:
                assignments = torch.ones(B, T, device=x.device, dtype=torch.long) * (self.n_recursions - 1)
                # Reset bias during warmup
                if self.balancing == "loss_free" and hasattr(self, "router_bias"):
                    self.router_bias.data.zero_()
            else:
                _, top_expert = torch.topk(router_probs_biased, 1, dim=-1, sorted=False)  # (B, T, 1)
                assignments = top_expert.squeeze(-1)             # (B, T)

            # Gate weights from the un-biased probs (for weighted gating)
            gate_scores = torch.gather(router_probs, dim=-1, index=assignments.unsqueeze(-1)).squeeze(-1)  # (B, T)

            self._assignments = assignments
            self._gate_scores_full = router_probs
            self._gate_weights = gate_scores                    # (B, T) — weight per token
            self._raw_logits  = raw_logits

        # ── Select tokens assigned to depth >= recursion_idx ──
        active_mask = (self._assignments >= recursion_idx)       # (B, T)

        # Convert mask to padded indices
        selected_list = []
        gate_list     = []
        max_k = max(1, int(active_mask.sum(dim=-1).max().item()))

        for b in range(B):
            idx = active_mask[b].nonzero(as_tuple=False).view(-1, 1)   # (k_b, 1)
            gv  = self._gate_weights[b][active_mask[b]]                # (k_b,)
            # Edge case: no tokens selected (decode token assigned to shallower depth)
            if idx.shape[0] == 0:
                idx = torch.zeros(1, 1, dtype=torch.long, device=x.device)
                gv  = torch.zeros(1, device=x.device)
            pad = max_k - idx.shape[0]
            if pad > 0:
                idx = torch.cat([idx, idx[-1:].expand(pad, 1)], dim=0)
                gv  = torch.cat([gv, gv[-1:].expand(pad)], dim=0)
            selected_list.append(idx)
            gate_list.append(gv)

        selected     = torch.stack(selected_list, dim=0)                # (B, max_k, 1)
        gate_weights = torch.stack(gate_list, dim=0).unsqueeze(-1)      # (B, max_k, 1)

        # ── Losses (only computed once at recursion_idx == 0) ──
        aux_loss = None
        z_loss   = None

        if self.training and recursion_idx == 0:
            balancing_ratio = torch.bincount(
                self._assignments.view(-1), minlength=self.n_recursions
            ).float() / (B * T)

            if self.balancing == "loss":
                # Reference-matching balancing loss: Σ P_i * f_i
                P_i = torch.sum(self._gate_scores_full, dim=(0, 1)) / (B * T)  # (N_r,)
                f_i = self.n_recursions * balancing_ratio
                aux_loss = self.balancing_loss_weight * (P_i * f_i).sum()

            elif self.balancing == "loss_free":
                # No gradient loss; router_bias is updated externally
                aux_loss = None

            if self.z_loss_weight > 0:
                z = torch.logsumexp(self._raw_logits, dim=-1)           # (B, T)
                z_loss = self.z_loss_weight * z.pow(2).mean()

        return RouterOutput(
            selected_tokens=selected,
            gate_weights=gate_weights,
            router_logits=self._raw_logits[:, :, :1] if self._raw_logits is not None else x[:, :, :1],
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
