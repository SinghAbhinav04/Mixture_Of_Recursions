#!/usr/bin/env bash
# scripts/train_small.sh
# ──────────────────────
# Train a ~85M (effective) / ~28M (unique) MoR model.
# Requires a CUDA GPU; estimated ~30 min on A100, ~2h on T4.

set -e

cd "$(dirname "$0")/.."

python train.py \
    --preset small \
    --dataset tiny_shakespeare \
    --max_steps 20000 \
    --batch_size 32 \
    --grad_accum 4 \
    --precision bf16 \
    --eval_interval 500 \
    --save_interval 2000 \
    --out_dir checkpoints/small_run \
    "$@"

echo ""
echo "✅ Done! Checkpoint in checkpoints/small_run/"
echo "   Run: python chat.py --checkpoint checkpoints/small_run/last.pt"
