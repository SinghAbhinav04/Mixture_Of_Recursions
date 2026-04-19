"""
generate.py
───────────
Standalone inference script for Abhinav-MoR.
Loads a checkpoint and generates text from a prompt.

Usage:
  python generate.py --ckpt /path/to/model.pt --prompt "Once upon a time"
"""
import os
import argparse
import torch
from model.mor_model import MoRForCausalLM
from data.dataset import decode, encode

def main():
    parser = argparse.ArgumentParser(description="Abhinav-MoR Text Generation")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--prompt", type=str, default="The future of AI is", help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    args = parser.parse_args()

    # ── 1. Setup Device ──
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint from {args.ckpt} to {device}...")

    # ── 2. Load Checkpoint ──
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    
    # Extract configs saved during training
    model_cfg = ckpt["model_cfg"]
    mor_cfg   = ckpt["mor_cfg"]
    step      = ckpt.get("step", "unknown")

    print(f"\nModel Architecture:")
    print(f"  Virtual Layers: {model_cfg.n_layers}")
    print(f"  Recursions:     {mor_cfg.n_recursions}")
    print(f"  Shared Across:  {mor_cfg.strategy}")
    print(f"  Trained for:    {step} steps")

    # ── 3. Build & Load Model ──
    model = MoRForCausalLM(model_cfg, mor_cfg)
    
    # Handle potentially compiled models (checkpoints might have '_orig_mod.' prefix)
    state_dict = ckpt["model_state_dict"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # ── 4. Tokenize Prompt ──
    print(f"\nPrompt: \"{args.prompt}\"")
    print("-" * 50)
    
    prompt_ids = torch.tensor(encode(args.prompt), dtype=torch.long, device=device).unsqueeze(0)

    # ── 5. Generate ──
    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids     = prompt_ids,
            max_new_tokens = args.max_new_tokens,
            temperature    = args.temperature,
            top_k          = args.top_k
        )

    # ── 6. Decode & Print ──
    result_text = decode(generated_ids[0].tolist())
    print(result_text)
    print("-" * 50)
    print("\nGeneration complete.")

if __name__ == "__main__":
    main()
