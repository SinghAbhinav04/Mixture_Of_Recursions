"""
train/scheduler.py
──────────────────
Learning rate scheduler: cosine decay with linear warmup.

Implementation follows the classic Chinchilla / LLaMA training recipe:
  • Linear warmup for `warmup_steps` steps
  • Cosine decay from `lr` to `min_lr` = `lr * min_lr_ratio`
  • Optional: flat LR after cosine reaches min_lr
"""
from __future__ import annotations
import math

import torch.optim as optim


def get_lr(
    step:          int,
    max_lr:        float,
    min_lr_ratio:  float = 0.1,
    warmup_steps:  int   = 200,
    max_steps:     int   = 5000,
) -> float:
    """
    Compute the learning rate at a given training step.

    Schedule:
      step < warmup_steps  → linear warmup from 0 to max_lr
      warmup ≤ step ≤ max  → cosine decay from max_lr to min_lr
      step > max_steps     → flat at min_lr

    Args:
        step:         current training step (0-indexed)
        max_lr:       peak learning rate
        min_lr_ratio: minimum LR as fraction of max_lr (default 0.1 = 10×)
        warmup_steps: linear warmup duration
        max_steps:    total steps for cosine decay

    Returns:
        Learning rate scalar
    """
    min_lr = max_lr * min_lr_ratio

    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Plateau after max_steps
    if step >= max_steps:
        return min_lr

    # Cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff       = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + (max_lr - min_lr) * coeff


class CosineWithWarmup:
    """
    Thin wrapper so you can use the scheduler with a torch optimizer
    via `.step()` calls, same as torch.optim.lr_scheduler.

    Usage:
        scheduler = CosineWithWarmup(optimizer, max_lr=3e-4,
                                     warmup_steps=200, max_steps=5000)
        ...
        # inside training loop:
        scheduler.step()
    """
    def __init__(
        self,
        optimizer:    optim.Optimizer,
        max_lr:       float,
        warmup_steps: int,
        max_steps:    int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer    = optimizer
        self.max_lr       = max_lr
        self.warmup_steps = warmup_steps
        self.max_steps    = max_steps
        self.min_lr_ratio = min_lr_ratio
        self._step        = 0

    def step(self):
        lr = get_lr(
            self._step,
            self.max_lr,
            self.min_lr_ratio,
            self.warmup_steps,
            self.max_steps,
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self._step += 1
        return lr

    def get_last_lr(self) -> float:
        return get_lr(
            max(0, self._step - 1),
            self.max_lr,
            self.min_lr_ratio,
            self.warmup_steps,
            self.max_steps,
        )
