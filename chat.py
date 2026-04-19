"""
chat.py
───────
Interactive chat interface for Abhinav-MoR.
Allows back-and-forth conversation with context memory.

Usage:
  python chat.py --ckpt /path/to/model.pt
"""
import os
import argparse
import torch
from model.mor_model import MoRForCausalLM
from data.dataset import decode, encode

def main():
    parser = argparse.ArgumentParser(description="Abhinav-MoR Interactive Chat")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    args = parser.parse_args()

    # ── 1. Setup Device ──
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[System] Loading model onto {device}...")

    # ── 2. Load Checkpoint ──
    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"[Error] Checkpoint not found at {args.ckpt}")
        return

    model_cfg = ckpt["model_cfg"]
    mor_cfg   = ckpt["mor_cfg"]
    
    # Reconstruct model
    model = MoRForCausalLM(model_cfg, mor_cfg)
    
    # Fix state_dict keys if they have '_orig_mod.' prefix (from torch.compile)
    state_dict = ckpt["model_state_dict"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"[System] Model Loaded. (Recursions: {mor_cfg.n_recursions}, Strategy: {mor_cfg.strategy})")
    print("-" * 60)
    print("Welcome to Abhinav-MoR Chat!")
    print("Type your message and press Enter. Type 'exit' or 'quit' to stop.")
    print("Type 'clear' to reset conversation memory.")
    print("-" * 60)

    # ── 3. Chat Logic ──
    history_ids = None

    while True:
        try:
            user_input = input("\nUser: ").strip()
        except EOFError:
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            history_ids = None
            print("[System] Conversation memory cleared.")
            continue

        # Add user input to history
        new_ids = torch.tensor(encode(user_input + "\n"), dtype=torch.long, device=device).unsqueeze(0)
        
        if history_ids is None:
            history_ids = new_ids
        else:
            history_ids = torch.cat([history_ids, new_ids], dim=1)
            
        # Ensure we don't exceed max_seq_len
        if history_ids.size(1) > model_cfg.max_seq_len - args.max_new_tokens:
            history_ids = history_ids[:, -(model_cfg.max_seq_len - args.max_new_tokens):]

        # Generate response
        print("MoR: ", end="", flush=True)
        
        with torch.no_grad():
            # Basic generation (non-streaming for reliability, but prints once)
            # You can implement token-by-token printing here if desired.
            generated_ids = model.generate(
                prompt_ids     = history_ids,
                max_new_tokens = args.max_new_tokens,
                temperature    = args.temperature,
                top_k          = args.top_k
            )
            
            # Extract only the NEW tokens
            response_ids = generated_ids[0, history_ids.size(1):]
            response_text = decode(response_ids.tolist())
            
            # Print the response
            print(response_text)
            
            # Update history with the model's response
            history_ids = generated_ids

if __name__ == "__main__":
    main()
