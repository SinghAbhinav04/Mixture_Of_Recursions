"""
train/trainer.py
────────────────
MoR-aware training loop with memory optimisations for constrained GPUs.

Memory fixes vs original:
  ① Dataset stays on CPU — only batches moved to GPU
  ② torch.cuda.empty_cache() after eval / checkpoint
  ③ Gradient checkpointing support (trades ~25% speed for ~60% activation mem)
  ④ GPU memory tracking in logs
  ⑤ Early OOM warning before training starts
"""
from __future__ import annotations
import gc
import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn

from model.mor_model import MoRForCausalLM
from data.dataset import get_batch
from train.scheduler import CosineWithWarmup
from configs.default import TrainConfig, ModelConfig


# ─────────────────────────────────────────────────────────────────
#  Loss Accumulator
# ─────────────────────────────────────────────────────────────────

class AverageMeter:
    def __init__(self):
        self.val   = 0.0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val   += val * n
        self.count += n

    def avg(self) -> float:
        return self.val / self.count if self.count > 0 else 0.0

    def reset(self):
        self.val   = 0.0
        self.count = 0


# ─────────────────────────────────────────────────────────────────
#  GPU Memory Utilities
# ─────────────────────────────────────────────────────────────────

def gpu_mem_used() -> float:
    """Return GPU memory in use (GB) or 0 if no CUDA."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e9
    return 0.0


def gpu_mem_reserved() -> float:
    """Return GPU memory reserved by allocator (GB) or 0 if no CUDA."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1e9
    return 0.0


def free_gpu_cache():
    """Release cached GPU memory back to OS."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def estimate_peak_mem(model: MoRForCausalLM, batch_size: int, seq_len: int) -> float:
    """
    Rough estimate of peak GPU memory (GB) during training.
    Helps warn users before OOM.
    """
    p = model.num_params()
    param_gb = p["unique"] * 2 / 1e9           # bf16 params
    optim_gb = p["unique"] * 8 / 1e9            # fp32 master + momentum + variance
    grad_gb  = p["unique"] * 2 / 1e9            # bf16 grads
    # Rough activation estimate: B * T * d * n_layers * 2 bytes * ~8 (residuals)
    act_gb = batch_size * seq_len * model.model_cfg.d_model * model.model_cfg.n_layers * 16 / 1e9
    # Logits: B * T * vocab * 2 bytes (bf16) — chunked CE avoids fp32 upcast
    logits_gb = batch_size * seq_len * model.model_cfg.vocab_size * 2 / 1e9
    return param_gb + optim_gb + grad_gb + act_gb + logits_gb


# ─────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:      MoRForCausalLM,
    val_data:   torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device:     str,
) -> Dict[str, float]:
    """
    Average loss over `eval_iters` random batches from val_data.

    Returns:
        dict with 'val_loss', 'val_lm_loss', 'val_ppl' (perplexity)
    """
    model.eval()
    lm_losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(val_data, block_size, batch_size, device=device)
        out  = model(x, y)
        lm_losses[k] = out.lm_loss.item()
    model.train()
    free_gpu_cache()
    avg_lm_loss = lm_losses.mean().item()
    return {
        "val_loss":    avg_lm_loss,
        "val_lm_loss": avg_lm_loss,
        "val_ppl":     float(torch.exp(torch.tensor(avg_lm_loss)).item()),
    }


# ─────────────────────────────────────────────────────────────────
#  Checkpoint Utilities
# ─────────────────────────────────────────────────────────────────

def save_checkpoint(
    path:      str,
    step:      int,
    model:     MoRForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineWithWarmup,
    metrics:   Dict[str, float],
    model_cfg: ModelConfig,
    mor_cfg,
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "step":                step,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_step":      scheduler._step,
        "metrics":             metrics,
        "model_cfg":           model_cfg,
        "mor_cfg":             mor_cfg,
    }, path)
    print(f"  ✓ Checkpoint saved → {path}")
    free_gpu_cache()


def load_checkpoint(path: str, model: MoRForCausalLM, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler._step = ckpt.get("scheduler_step", 0)
    step = ckpt["step"]
    print(f"  ✓ Resumed from step {step}")
    return step


# ─────────────────────────────────────────────────────────────────
#  Main Training Loop
# ─────────────────────────────────────────────────────────────────

def train(
    model:      MoRForCausalLM,
    train_data: torch.Tensor,
    val_data:   torch.Tensor,
    cfg:        TrainConfig,
    resume_from: Optional[str] = None,
) -> MoRForCausalLM:
    """
    Full MoR training loop with memory optimisations.

    Args:
        model:       MoRForCausalLM instance (already moved to device)
        train_data:  1D LongTensor of training token IDs (CPU is fine)
        val_data:    1D LongTensor of validation token IDs (CPU is fine)
        cfg:         TrainConfig
        resume_from: optional path to a checkpoint to resume from

    Returns:
        Trained model
    """
    # ── Device ──
    device = cfg.device or ("cuda" if torch.cuda.is_available() else
                             "mps"  if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)

    # ── Early OOM warning ──
    if device == "cuda":
        total_vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        estimated = estimate_peak_mem(model, cfg.batch_size, model.model_cfg.max_seq_len)
        if estimated > total_vram * 0.9:
            print(f"\n  ⚠ WARNING: Estimated peak memory {estimated:.1f} GB "
                  f"exceeds GPU VRAM {total_vram:.1f} GB!")
            print(f"    → Try: --preset kaggle_small  OR  --batch_size 4 --grad_accum 32")
            print(f"    → Or:  --gradient_checkpointing (already on by default)\n")

    # ── Mixed precision scaler ──
    use_amp   = cfg.precision in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float16

    try:
        scaler = torch.amp.GradScaler(device_type="cuda", enabled=(cfg.precision == "fp16" and torch.cuda.is_available()))
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(cfg.precision == "fp16" and torch.cuda.is_available()))

    # ── Compile (PyTorch 2.0+) ──
    if cfg.compile:
        print("Compiling model with torch.compile() ...")
        model = torch.compile(model)

    # ── Optimizer ──
    decay_params = [p for n, p in model.named_parameters() if p.ndim >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,   "weight_decay": cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        fused=torch.cuda.is_available(),
    )

    # ── Scheduler ──
    scheduler = CosineWithWarmup(
        optimizer,
        max_lr       = cfg.lr,
        warmup_steps = cfg.warmup_steps,
        max_steps    = cfg.max_steps,
    )

    start_step = 0
    if resume_from and os.path.isfile(resume_from):
        start_step = load_checkpoint(resume_from, model, optimizer, scheduler)

    # ── Loss meters ──
    meters = {k: AverageMeter() for k in ("total", "lm", "aux", "z")}

    model.train()
    t0 = time.time()

    block_size = model.model_cfg.max_seq_len
    out_dir    = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training Abhinav-MoR")
    print(f"  Device: {device}  |  Precision: {cfg.precision}")
    print(f"  Steps: {cfg.max_steps}  |  Batch: {cfg.batch_size}  |  Grad accum: {cfg.grad_accum}")
    print(f"  Gradient checkpointing: {cfg.gradient_checkpointing}")
    if device == "cuda":
        print(f"  GPU VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print(f"{'='*60}\n")

    for step in range(start_step, cfg.max_steps):

        # ── Gradient accumulation micro-steps ──
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.tensor(0.0, device=device)

        for micro_step in range(cfg.grad_accum):
            x, y = get_batch(train_data, block_size, cfg.batch_size, device=device)

            with torch.autocast(device_type=device.split(":")[0],
                                dtype=amp_dtype, enabled=use_amp):
                out = model(x, y)

            loss = out.loss / cfg.grad_accum
            scaler.scale(loss).backward()
            total_loss += loss.detach()

            if out.lm_loss is not None:
                meters["lm"].update(out.lm_loss.item())
            if out.aux_loss is not None:
                meters["aux"].update(out.aux_loss.item())
            if out.z_loss is not None:
                meters["z"].update(out.z_loss.item())

            del out, loss

        # ── Gradient clip + update ──
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        lr = scheduler.step()
        meters["total"].update(total_loss.item())
        del total_loss

        # ── Logging ──
        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            tokens_per_sec = (cfg.batch_size * block_size * cfg.grad_accum * cfg.log_interval) / dt
            mem_str = f" | gpu {gpu_mem_used():.1f}/{gpu_mem_reserved():.1f} GB" if device == "cuda" else ""
            print(
                f"step {step:6d} | "
                f"loss {meters['total'].avg():.4f} | "
                f"lm {meters['lm'].avg():.4f} | "
                f"aux {meters['aux'].avg():.5f} | "
                f"z {meters['z'].avg():.5f} | "
                f"lr {lr:.2e} | "
                f"tok/s {tokens_per_sec:,.0f}"
                f"{mem_str}"
            )
            for m in meters.values():
                m.reset()

        # ── Evaluation ──
        if step % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            metrics = evaluate(model, val_data, block_size, cfg.batch_size, cfg.eval_iters, device)
            print(
                f"\n  ── Eval @ step {step} ──\n"
                f"  val_loss={metrics['val_loss']:.4f}  "
                f"val_ppl={metrics['val_ppl']:.2f}\n"
            )

        # ── Checkpoint ──
        if (step % cfg.save_interval == 0 and step > 0) or step == cfg.max_steps - 1:
            ckpt_path = os.path.join(out_dir, f"step_{step:06d}.pt")
            save_checkpoint(
                path      = ckpt_path,
                step      = step,
                model     = model,
                optimizer = optimizer,
                scheduler = scheduler,
                metrics   = {},
                model_cfg = model.model_cfg,
                mor_cfg   = model.mor_cfg,
            )
            import shutil
            last_path = os.path.join(out_dir, "last.pt")
            shutil.copy2(ckpt_path, last_path)

    print(f"\n✅ Training complete! Model saved to {out_dir}/last.pt")
    return model
