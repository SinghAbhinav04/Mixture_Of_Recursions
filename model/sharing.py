"""
model/sharing.py
────────────────
Parameter-sharing strategy functions.

Maps a virtual layer index ℓ ∈ [0, L) to a unique block index,
implementing the core parameter reuse mechanism of MoR.

Strategy overview (L=12, N_r=3 example):
  cycle:           0 1 2 3 0 1 2 3 0 1 2 3   (pure round-robin)
  middle_cycle:    0 1 2 3 1 2 3 1 2 3 4 5   (first/last unique, recommended)
  sequence:        0 0 0 0 1 1 1 1 2 2 2 2   (consecutive groups)
  middle_sequence: 0 1 1 1 1 2 2 2 2 3 3 4   (sequence + pinned ends)
"""
from __future__ import annotations
from typing import Callable, Dict


# ─────────────────────────────────────────────────────────────────
#  Strategy Functions: ℓ → block_index
# ─────────────────────────────────────────────────────────────────

def cycle(ell: int, L: int, N_r: int) -> int:
    """
    Cycle: Φ'_{ℓ mod (L / N_r)}
    Pure round-robin — every N_r-th layer reuses a block.
    """
    return ell % (L // N_r)


def middle_cycle(ell: int, L: int, N_r: int) -> int:
    """Middle-Cycle (recommended): first/last layers get unique blocks, middle layers cycle."""
    if ell == 0:
        return 0
    if ell == L - 1:
        return N_r + 1
    inner_L = L - 2
    period  = inner_L // N_r
    return ((ell - 1) % period) + 1


def sequence(ell: int, L: int, N_r: int) -> int:
    """
    Sequence: Φ'_{⌊ℓ / N_r⌋}
    Consecutive N_r layers share one block.
    """
    return ell // N_r


def middle_sequence(ell: int, L: int, N_r: int) -> int:
    """
    Middle-Sequence:
      - Layer 0   → unique block 0
      - Layer L-1 → unique block (last)
      - Middle    → consecutive groups of N_r
    """
    if ell == 0:
        return 0
    if ell == L - 1:
        n_unique = (L - 2) // N_r + 2
        return n_unique - 1
    return ((ell - 1) // N_r) + 1


# ─────────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────────

SHARING_STRATEGIES: Dict[str, Callable[[int, int, int], int]] = {
    "cycle":           cycle,
    "middle_cycle":    middle_cycle,
    "sequence":        sequence,
    "middle_sequence": middle_sequence,
}


# ─────────────────────────────────────────────────────────────────
#  Utility: count unique blocks & build routing table
# ─────────────────────────────────────────────────────────────────

def count_unique_blocks(strategy: str, L: int, N_r: int) -> int:
    """Return the number of unique TransformerBlocks required."""
    fn = SHARING_STRATEGIES[strategy]
    return len({fn(ell, L, N_r) for ell in range(L)})


def build_routing_table(strategy: str, L: int, N_r: int) -> list[int]:
    """
    Build a length-L list: routing_table[ℓ] = shared block index.

    Example (L=12, N_r=3, strategy='middle_cycle'):
        [0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5]
    """
    fn = SHARING_STRATEGIES[strategy]
    return [fn(ell, L, N_r) for ell in range(L)]


def print_routing_table(strategy: str, L: int, N_r: int) -> None:
    """Pretty-print the routing table for quick sanity checks."""
    table = build_routing_table(strategy, L, N_r)
    n_unique = count_unique_blocks(strategy, L, N_r)
    print(f"Strategy={strategy!r}  L={L}  N_r={N_r}  unique_blocks={n_unique}")
    print(f"Routing: {table}")
