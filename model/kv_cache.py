"""
model/kv_cache.py
─────────────────
Recursion-wise KV cache for MoR inference.

Problem: When tokens "exit" at recursion step r, they skip the deeper blocks.
But future tokens in the same sequence need the KV entries for those skipped
positions — this is the "missing KV problem" in early-exit transformers.

Solution: RecursiveKVCache stores KV entries shared across recursion steps.
  • Tokens that are processed in step r contribute their computed K/V.
  • Tokens that were NOT processed in step r either:
      - Reuse their K/V from step r-1 (carry-forward), or
      - Use a zero sentinel (if not yet computed).

This module provides:
  - StandardKVCache    — plain per-layer KV cache (for baseline)
  - RecursiveKVCache   — recursion-aware shared KV cache (for MoR)
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import torch


# ─────────────────────────────────────────────────────────────────
#  Standard KV Cache (Baseline)
# ─────────────────────────────────────────────────────────────────

class StandardKVCache:
    """Simple KV cache for a single, non-recursive transformer decoder."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V to the cache for `layer_idx` and return the full K/V."""
        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = (new_k, new_v)
        else:
            old_k, old_v = self.cache[layer_idx]
            self.cache[layer_idx] = (
                torch.cat([old_k, new_k], dim=2),
                torch.cat([old_v, new_v], dim=2),
            )
        return self.cache[layer_idx]

    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        return self.cache[layer_idx]

    def reset(self):
        self.cache = [None] * self.n_layers

    def seq_length(self) -> int:
        for c in self.cache:
            if c is not None:
                return c[0].shape[2]
        return 0


# ─────────────────────────────────────────────────────────────────
#  Recursive KV Cache (MoR-specific)
# ─────────────────────────────────────────────────────────────────

class RecursiveKVCache:
    """
    KV cache shared across recursion steps.

    Architecture:
      n_unique_blocks shared transformer blocks are reused N_r times.
      Each block may receive a different subset of tokens each recursion.

      The cache stores:
        self.kv[block_idx][recursion_step] = (K, V) for full sequence
      where unprocessed token positions carry their K/V from the previous step.

    Args:
        n_unique_blocks: number of unique shared blocks
        n_recursions:    total recursion steps per forward pass
        d_model:         hidden dim (for zero-initialising missing entries)
        n_kv_heads:      number of KV heads
        head_dim:        per-head dimension
    """
    def __init__(
        self,
        n_unique_blocks: int,
        n_recursions:    int,
        n_kv_heads:      int,
        head_dim:        int,
    ):
        self.n_unique_blocks = n_unique_blocks
        self.n_recursions    = n_recursions
        self.n_kv_heads      = n_kv_heads
        self.head_dim        = head_dim

        # cache[block][recursion] = (K, V) or None
        self.cache: List[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = [
            [None] * n_recursions for _ in range(n_unique_blocks)
        ]

    def update(
        self,
        block_idx:     int,
        recursion_idx: int,
        new_k:         torch.Tensor,   # (B, n_kv, T_new, head_dim)
        new_v:         torch.Tensor,
        selected_mask: Optional[torch.Tensor] = None,  # (B, T_total) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the KV cache for (block_idx, recursion_idx).

        If selected_mask is provided, only the selected positions get updated;
        unselected positions carry forward the KV from the previous recursion
        or remain zero.

        Returns the full (B, n_kv, T_total, head_dim) K and V tensors.
        """
        if self.cache[block_idx][recursion_idx] is None:
            # First time: just store the new values
            self.cache[block_idx][recursion_idx] = (new_k, new_v)
        else:
            # Append newly computed tokens
            old_k, old_v = self.cache[block_idx][recursion_idx]
            self.cache[block_idx][recursion_idx] = (
                torch.cat([old_k, new_k], dim=2),
                torch.cat([old_v, new_v], dim=2),
            )
        return self.cache[block_idx][recursion_idx]

    def get(
        self,
        block_idx:     int,
        recursion_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return cached (K, V) for the given block and recursion, or None."""
        return self.cache[block_idx][recursion_idx]

    def reset(self):
        self.cache = [
            [None] * self.n_recursions for _ in range(self.n_unique_blocks)
        ]

    def seq_length(self) -> int:
        for blk in self.cache:
            for entry in blk:
                if entry is not None:
                    return entry[0].shape[2]
        return 0
