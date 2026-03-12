"""
eval/evaluate.py
────────────────
Evaluation utilities for MoR.

Metrics:
  • Validation cross-entropy loss (per-token)
  • Perplexity (exp of loss)
  • Router statistics: average selected fraction per recursion step

Usage:
    python eval/evaluate.py --checkpoint checkpoints/last.pt --dataset tiny_shakespeare
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from model.mor_model import MoRForCausalLM
from data.dataset import get_dataset, get_batch


# ─────────────────────────────────────────────────────────────────
#  Core evaluation function
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model:      MoRForCausalLM,
    val_data:   torch.Tensor,
    block_size: int,
    batch_size: int = 16,
    eval_iters: int = 200,
    device:     str = "cpu",
) -> dict:
    """
    Evaluate model on validation data.

    Returns:
        dict with:
          val_loss   — average cross-entropy loss
          val_ppl    — perplexity (e^loss)
          lm_loss    — LM component only
          aux_loss   — router BCE component
          z_loss     — router z-loss component
    """
    model.eval()

    lm_losses   = []
    aux_losses  = []
    z_losses    = []
    # Router stats: fraction of tokens selected per recursion step
    router_fracs = [[] for _ in range(model.mor_cfg.n_recursions)]

    for _ in range(eval_iters):
        x, y = get_batch(val_data, block_size, batch_size, device=device)
        out  = model(x, y)

        if out.lm_loss is not None:
            lm_losses.append(out.lm_loss.item())
        if out.aux_loss is not None:
            aux_losses.append(out.aux_loss.item())
        if out.z_loss is not None:
            z_losses.append(out.z_loss.item())

    avg_lm  = sum(lm_losses)  / max(1, len(lm_losses))
    avg_aux = sum(aux_losses) / max(1, len(aux_losses))
    avg_z   = sum(z_losses)   / max(1, len(z_losses))

    model.train()

    return {
        "val_loss": avg_lm,
        "val_ppl":  float(torch.exp(torch.tensor(avg_lm))),
        "lm_loss":  avg_lm,
        "aux_loss": avg_aux,
        "z_loss":   avg_z,
    }


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate an Abhinav-MoR checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset",    default="tiny_shakespeare")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_iters", type=int, default=200)
    p.add_argument("--device",     default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    # Load checkpoint
    ckpt      = torch.load(args.checkpoint, map_location=device)
    model_cfg = ckpt["model_cfg"]
    mor_cfg   = ckpt["mor_cfg"]
    model     = MoRForCausalLM(model_cfg, mor_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model     = model.to(device)

    # Load val data
    _, val_data = get_dataset(args.dataset, device=device)

    # Evaluate
    results = evaluate_model(
        model,
        val_data,
        block_size = model_cfg.max_seq_len,
        batch_size = args.batch_size,
        eval_iters = args.eval_iters,
        device     = device,
    )

    print("\n" + "="*50)
    print("  Evaluation Results")
    print("="*50)
    for k, v in results.items():
        print(f"  {k:12s}: {v:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
