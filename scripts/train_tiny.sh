#!/usr/bin/env bash
# scripts/train_tiny.sh
# ─────────────────────
# Quick smoke test on TinyShakespeare (~1MB).
# TinyShakespeare downloads automatically.
# Should finish in ~2 minutes on a GPU, ~10 minutes on CPU.

set -e

cd "$(dirname "$0")/.."

python train.py \
    --dataset tiny_shakespeare \
    --d_model 256 \
    --n_layers 12 \
    --n_heads 4 \
    --n_kv_heads 4 \
    --max_seq_len 256 \
    --n_recursions 3 \
    --strategy middle_cycle \
    --routing_type expert \
    --max_steps 2000 \
    --batch_size 16 \
    --lr 3e-4 \
    --warmup_steps 100 \
    --eval_interval 200 \
    --save_interval 1000 \
    --out_dir checkpoints/tiny_run \
    "$@"

echo ""
echo "✅ Done! Checkpoint in checkpoints/tiny_run/"
echo "   Run: python chat.py --checkpoint checkpoints/tiny_run/last.pt"
