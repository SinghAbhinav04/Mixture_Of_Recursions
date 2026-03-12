#!/usr/bin/env python3
"""
chat.py — Interactive inference with a trained Abhinav-MoR model.

Usage:
    # Interactive chat (prompt-continue mode):
    python chat.py --checkpoint checkpoints/last.pt

    # Single generation from CLI:
    python chat.py --checkpoint checkpoints/last.pt \\
        --prompt "To be or not to be" \\
        --max_new_tokens 200 --temperature 0.8 --top_k 50
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch

from model.mor_model import MoRForCausalLM
from data.dataset import encode, decode


def load_model(checkpoint_path: str, device: str) -> MoRForCausalLM:
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_cfg = ckpt["model_cfg"]
    mor_cfg   = ckpt["mor_cfg"]

    model = MoRForCausalLM(model_cfg, mor_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    p = model.num_params()
    print(f"Model loaded: {p['total']/1e6:.2f}M params ({p['unique']/1e6:.2f}M unique)")
    return model


def generate_text(
    model:          MoRForCausalLM,
    prompt:         str,
    max_new_tokens: int   = 200,
    temperature:    float = 0.8,
    top_k:          int   = 50,
    top_p:          float = None,
    device:         str   = "cpu",
) -> str:
    ids      = encode(prompt)
    idx      = torch.tensor([ids], dtype=torch.long, device=device)
    out_idx  = model.generate(
        idx,
        max_new_tokens = max_new_tokens,
        temperature    = temperature,
        top_k          = top_k,
        top_p          = top_p,
    )
    return decode(out_idx[0].tolist())


def parse_args():
    p = argparse.ArgumentParser(description="Chat with a trained Abhinav-MoR model.")
    p.add_argument("--checkpoint",      required=True, help="Path to .pt checkpoint")
    p.add_argument("--prompt",          default=None,  help="If set, generate once and exit")
    p.add_argument("--max_new_tokens",  type=int,   default=200)
    p.add_argument("--temperature",     type=float, default=0.8)
    p.add_argument("--top_k",           type=int,   default=50)
    p.add_argument("--top_p",           type=float, default=None)
    p.add_argument("--device",          default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    model = load_model(args.checkpoint, device)

    # ── Single prompt mode ──
    if args.prompt:
        output = generate_text(
            model, args.prompt,
            max_new_tokens = args.max_new_tokens,
            temperature    = args.temperature,
            top_k          = args.top_k,
            top_p          = args.top_p,
            device         = device,
        )
        print(output)
        return

    # ── Interactive chat loop ──
    print("\n" + "="*60)
    print("  Abhinav-MoR — Interactive Mode")
    print(f"  Device: {device}")
    print("  Type your prompt and press Enter. 'quit' to exit.")
    print("="*60)

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not prompt:
            print("(empty prompt — try again)")
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        output = generate_text(
            model, prompt,
            max_new_tokens = args.max_new_tokens,
            temperature    = args.temperature,
            top_k          = args.top_k,
            top_p          = args.top_p,
            device         = device,
        )
        print("\nMoR:")
        print("-" * 40)
        print(output)
        print("-" * 40)


if __name__ == "__main__":
    main()
