#!/usr/bin/env python3
"""
train.py — single training entrypoint.

Usage:
    # Quick test on TinyShakespeare (tiny model, ~30 seconds on GPU):
    python train.py

    # Using a named preset:
    python train.py --preset small --max_steps 20000

    # Full fine-grained control:
    python train.py \\
        --dataset tiny_shakespeare \\
        --d_model 512 --n_layers 12 --n_heads 8 --n_kv_heads 4 \\
        --n_recursions 3 --strategy middle_cycle \\
        --max_steps 5000 --batch_size 32 --lr 3e-4 \\
        --out_dir checkpoints/my_run

    # Resume from checkpoint:
    python train.py --resume checkpoints/my_run/last.pt
"""
import argparse
import sys
import os

# ── make sure imports work when running from project root ──
sys.path.insert(0, os.path.dirname(__file__))

import torch

from configs.default import ModelConfig, MoRConfig, RouterConfig, TrainConfig, PRESETS
from model.mor_model import MoRForCausalLM
from data.dataset import get_dataset
from train.trainer import train


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Abhinav-MoR: Train a Mixture-of-Recursions language model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Preset ---
    p.add_argument("--preset", choices=list(PRESETS), default=None,
                   help="Named preset config (tiny/small/medium). Overrides individual flags.")

    # --- Dataset ---
    p.add_argument("--dataset", default="tiny_shakespeare",
                   help="'tiny_shakespeare' | 'fineweb' | 'openwebtext' | /path/to/file.txt")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit HuggingFace dataset to N samples")

    # --- Model architecture ---
    p.add_argument("--d_model",     type=int, default=256)
    p.add_argument("--n_layers",    type=int, default=12)
    p.add_argument("--n_heads",     type=int, default=4)
    p.add_argument("--n_kv_heads",  type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--dropout",     type=float, default=0.1)

    # --- MoR hyperparameters ---
    p.add_argument("--n_recursions", type=int,   default=3)
    p.add_argument("--strategy",     type=str,   default="middle_cycle",
                   choices=["cycle", "middle_cycle", "sequence", "middle_sequence"])
    p.add_argument("--routing_type", type=str,   default="expert",
                   choices=["expert", "token"])
    p.add_argument("--alpha",        type=float, default=0.1,
                   help="Gate weight scale factor")
    p.add_argument("--aux_loss_w",   type=float, default=0.001,
                   help="BCE auxiliary loss weight")
    p.add_argument("--z_loss_w",     type=float, default=1e-3,
                   help="Z-loss weight for router stability")
    p.add_argument("--cap_warmup",   type=int,   default=1000,
                   help="Capacity warmup steps")
    p.add_argument("--no_kv_sharing", action="store_true",
                   help="Disable KV sharing across recursion steps")

    # --- Training ---
    p.add_argument("--max_steps",    type=int,   default=2000)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--grad_accum",   type=int,   default=1)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int,   default=200)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--precision",    type=str,   default="bf16",
                   choices=["fp32", "fp16", "bf16"])
    p.add_argument("--compile",      action="store_true",
                   help="torch.compile() the model (PyTorch 2.0+)")
    p.add_argument("--seed",         type=int,   default=42)

    # --- I/O ---
    p.add_argument("--out_dir",      type=str,   default="checkpoints")
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--eval_interval",type=int,   default=250)
    p.add_argument("--save_interval",type=int,   default=1000)
    p.add_argument("--log_interval", type=int,   default=10)
    p.add_argument("--device",       type=str,   default=None,
                   help="Force device: 'cuda', 'mps', 'cpu'")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    # ── Auto-detect device ──
    device = args.device or (
        "cuda"  if torch.cuda.is_available() else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}")

    # ── Build configs ──
    if args.preset:
        model_cfg, mor_cfg, train_cfg = PRESETS[args.preset]()
        print(f"Using preset: {args.preset}")

        # Override preset values with CLI args if specified
        # (Check sys.argv to see what the user actually passed)
        cli_args = sys.argv[1:]
        if "--dataset" in cli_args: train_cfg.dataset = args.dataset
        if "--max_steps" in cli_args: train_cfg.max_steps = args.max_steps
        if "--batch_size" in cli_args: train_cfg.batch_size = args.batch_size
        if "--lr" in cli_args: train_cfg.lr = args.lr
        if "--precision" in cli_args: train_cfg.precision = args.precision
        if "--compile" in cli_args: train_cfg.compile = args.compile
        if "--out_dir" in cli_args: train_cfg.out_dir = args.out_dir
        if "--device" in cli_args: train_cfg.device = args.device
    else:
        model_cfg = ModelConfig(
            d_model    = args.d_model,
            n_layers   = args.n_layers,
            n_heads    = args.n_heads,
            n_kv_heads = args.n_kv_heads,
            max_seq_len= args.max_seq_len,
            dropout    = args.dropout,
        )
        router_cfg = RouterConfig(
            alpha            = args.alpha,
            aux_loss_weight  = args.aux_loss_w,
            z_loss_weight    = args.z_loss_w,
            cap_warmup_steps = args.cap_warmup,
            router_type      = args.router_type,
        )
        mor_cfg = MoRConfig(
            n_recursions = args.n_recursions,
            strategy     = args.strategy,
            routing_type = args.routing_type,
            router       = router_cfg,
            kv_sharing   = not args.no_kv_sharing,
        )
        train_cfg = TrainConfig(
            dataset       = args.dataset,
            max_steps     = args.max_steps,
            batch_size    = args.batch_size,
            grad_accum    = args.grad_accum,
            lr            = args.lr,
            warmup_steps  = args.warmup_steps,
            weight_decay  = args.weight_decay,
            grad_clip     = args.grad_clip,
            precision     = args.precision,
            compile       = args.compile,
            seed          = args.seed,
            out_dir       = args.out_dir,
            eval_interval = args.eval_interval,
            save_interval = args.save_interval,
            log_interval  = args.log_interval,
            device        = device,
        )

    # ── Build model ──
    model = MoRForCausalLM(model_cfg, mor_cfg)
    model.print_summary()

    # ── Load data ──
    train_data, val_data = get_dataset(
        source      = train_cfg.dataset,
        max_samples = args.max_samples,
        device      = device,
    )

    # ── Train ──
    train(
        model       = model,
        train_data  = train_data,
        val_data    = val_data,
        cfg         = train_cfg,
        resume_from = args.resume,
    )


if __name__ == "__main__":
    main()
