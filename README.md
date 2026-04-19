# MoR 
### *Mixture-of-Recursions Language Model — Production Implementation*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Built by Abhinav Singh** — implemented using a combination of AI assistance and hand-written code. This system is inspired by the research paper *"Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation"* (Bae, Kim, Bayat et al., NeurIPS 2025, KAIST AI / Google DeepMind / Google Research), and extends it into a fully working, self-contained training and inference system.
>
> 📄 [Read the research paper](https://arxiv.org/abs/2507.10524)

---

## Overview

Most language models apply the **same depth of computation to every single token**, regardless of whether the token is trivial or complex. This is wasteful.

**MoR** solves this with a learnable routing mechanism that dynamically assigns each token a recursion depth — **easy tokens exit early**, **hard tokens get deeper processing** — all through shared transformer blocks that are reused adaptively:

```
Standard Transformer:
  Every token → [Layer 1] → [Layer 2] → ... → [Layer 12] → output
                ─────────────────────────────────────────────────
                All 12 layers for every token, regardless of difficulty

MoR:
  Token → Router → top-k? → [Shared Block Group A] ─────────────────→ scatter
        → Router → top-k? → [Shared Block Group A] (same weights!) → scatter
        → Router → top-k? → [Shared Block Group A] (recursion 3)   → scatter
        ─────────────────────────────────────────────────────────────────────
  ✓ Hard tokens: process all 3 recursion steps (full depth)
  ✓ Easy tokens: exit early after step 1 or 2 (less compute, faster)
  ✓ Shared weights: 3 recursions with 1× the unique parameters
```

This gives us up to **2× inference throughput** at equivalent accuracy, with significantly fewer unique trainable parameters than a standard stacked transformer.

---

## Research Foundation

This system is built on the architecture described in:

> **Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation**
> Sangmin Bae, Yujin Kim, Reza Bayat, Sungnyun Kim, Jiyoun Ha, Tal Schuster, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Aaron Courville, Se-Young Yun
> *NeurIPS 2025 · KAIST AI · Google DeepMind · Google Research · Mila*
> [arXiv:2507.10524](https://arxiv.org/abs/2507.10524)

Key ideas we implement and extend:
- **Parameter sharing** across recursion steps (4 strategies: cycle, middle_cycle, sequence, middle_sequence)
- **Expert-choice routing** — model selects top-k tokens per step
- **Token-choice routing** — tokens self-select via threshold gating
- **Recursion-wise KV caching** — solves the missing KV cache problem at inference
- **End-to-end router training** — BCE auxiliary loss + z-loss regularisation

---

## Architecture

### High-Level Design

```
MoRForCausalLM
│
├── [Embeddings]
│   └── tok_emb: Embedding(vocab → d_model)   + dropout
│
├── [Shared Transformer Blocks]           ← N_unique blocks (much fewer than N_layers)
│   └── TransformerBlock × N_unique
│       ├── RMSNorm → CausalSelfAttention (RoPE + GQA-ready)
│       └── RMSNorm → SwiGLUFFN
│
├── [MoR Recursion Loop]                  ← N_r steps, reusing shared blocks
│   ├── Step 0: ExpertChoiceMoRLayer (capacity 100%)
│   │   ├── LinearRouter → sigmoid scores → top-k selection
│   │   ├── gather(x, top_k_idx)          ← physical sequence reduction
│   │   ├── run shared blocks on reduced  ← real compute savings
│   │   └── scatter_add back with gating  ← gated residual merge
│   ├── Step 1: ExpertChoiceMoRLayer (capacity 75%)
│   └── Step 2: ExpertChoiceMoRLayer (capacity 50%)
│
├── [Output Head]
│   ├── RMSNorm (final)
│   └── lm_head: Linear (weights tied with tok_emb)
│
└── [Training Losses]
    ├── lm_loss:   cross-entropy next-token prediction
    ├── aux_loss:  BCE router supervision (router learns what to select)
    └── z_loss:    log-sum-exp penalty (router stability)
```

### Backbone: LLaMA-Style Decoder

Every `TransformerBlock` implements the modern LLM recipe:

| Component | Design Choice | Why |
|---|---|---|
| **Norm** | RMSNorm | Cheaper than LayerNorm, no mean subtraction |
| **Position** | RoPE (Rotary Embeddings) | Generalises to longer sequences, no learned positions |
| **FFN** | SwiGLU (`SiLU(W_gate·x) ⊙ W_up·x → W_down`) | Higher quality than GELU at same params |
| **Attention** | Grouped Query Attention (GQA) | Fewer KV heads → faster inference, less memory |
| **Bias** | None in linear layers | Follows LLaMA/Gemma best practices |
| **Tying** | `lm_head.weight = tok_emb.weight` | Reduces params, improves training |

### Parameter Sharing Strategies

The `n_layers` virtual transformer layers are mapped to `n_unique` shared blocks. Four strategies are supported (example: L=12, N_r=3):

| Strategy | Virtual → Block mapping | Unique blocks |
|---|---|---|
| `cycle` | `[0,1,2,3, 0,1,2,3, 0,1,2,3]` | 4 |
| `middle_cycle` | `[0, 1,2,3,1,2,3,1,2,3, 4,5]` | 6 ← **recommended** |
| `sequence` | `[0,0,0,0, 1,1,1,1, 2,2,2,2]` | 4 |
| `middle_sequence` | `[0, 1,1,1,1,2,2,2,2,3, 4,5]` | 6 |

`middle_cycle` gives unique first/last blocks and cycles the middle — best empirical balance.

### Expert-Choice Routing (Core Mechanism)

```python
# For recursion step r:
raw_logits  = W_router · x                     # (B, T, 1)
gate_values = sigmoid(raw_logits) * alpha      # (B, T, 1) ∈ [0, alpha]
top_k       = int(capacity_r * T)             # shrinks each recursion step

selected, gate_w = topk(gate_values, k=top_k) # physical token selection
selected = sort(selected)                      # CRITICAL: maintain causal order

# ─── Physical sequence reduction (the real speedup) ───
x_reduced = gather(x, selected)               # (B, T, d) → (B, k, d)
out_k     = TransformerBlocks(x_reduced)      # only k tokens through attention!

# ─── Gated residual merge ───
x = scatter_add(x, selected, out_k * gate_w)

# ─── Training losses ───
aux_loss = BCE(raw_logits, top_k_binary_mask) * aux_loss_weight
z_loss   = mean(log_sum_exp(raw_logits)²)    * z_loss_weight
```

### KV Cache Sharing

At inference, tokens that exit early (lower recursion depth) still need their KV entries available for future token generation. `RecursiveKVCache` stores KV pairs per `(block_idx, recursion_step)` and carries forward entries from earlier recursions — **solving the missing KV problem**.

---

## Installation

```bash
# Clone and enter the project
git clone <your-repo-url> abhinav-mor
cd abhinav-mor

# Install core dependencies
pip install torch tiktoken

# For large-scale datasets (FineWeb, OpenWebText, etc.)
pip install datasets transformers
```

**Python:** 3.10+ | **PyTorch:** 2.1+ | **CUDA:** Optional but recommended

---

## Quick Start

### Train on TinyShakespeare (2–5 min on GPU, ~30 min on CPU)

```bash
python train.py
```
That's it. TinyShakespeare (~1MB) downloads automatically.

### Use named presets

```bash
# Tiny  — ~3.5M unique params, runs on CPU
python train.py --preset tiny --max_steps 2000

# Small — ~28M unique params, needs GPU
python train.py --preset small --max_steps 20000

# Medium — ~120M unique params, A100/H100 recommended
python train.py --preset medium --dataset fineweb

# Large — ~1B unique params, 128k context window (Requires massive VRAM!)
python train.py --preset large --dataset openwebtext
```

### Full manual control

```bash
python train.py \
    --dataset tiny_shakespeare \
    --d_model 512 --n_layers 12 --n_heads 8 --n_kv_heads 4 \
    --n_recursions 3 --strategy middle_cycle \
    --routing_type expert --alpha 0.1 \
    --aux_loss_w 0.001 --z_loss_w 0.001 \
    --max_steps 10000 --batch_size 32 --grad_accum 4 \
    --lr 3e-4 --warmup_steps 500 \
    --precision bf16 \
    --out_dir checkpoints/my_run
```

### Resume training from checkpoint

```bash
python train.py --resume checkpoints/my_run/last.pt
```

### Generate / Chat

```bash
# Interactive chat mode:
python chat.py --checkpoint checkpoints/tiny_run/last.pt

# Single generation from CLI:
python chat.py --checkpoint checkpoints/tiny_run/last.pt \
    --prompt "To be or not to be" \
    --max_new_tokens 200 --temperature 0.8 --top_k 50

# With nucleus (top-p) sampling:
python chat.py --checkpoint checkpoints/tiny_run/last.pt \
    --prompt "Once upon a time" --top_p 0.9 --temperature 0.7
```

### Evaluate

```bash
python eval/evaluate.py --checkpoint checkpoints/tiny_run/last.pt --eval_iters 200
```

---

## Training Output

```
Strategy='middle_cycle'  L=12  N_r=3  unique_blocks=6
Routing: [0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5]
[MoRForCausalLM] capacities=[1.0, 0.667, 0.333]

============================================================
  MoR
  d_model:       256  |  n_layers: 12 virtual (6 unique, 3×)
  n_heads / kv:  4 Q  /  4 KV
  max_seq_len:   256  |  vocab: 200,019
  Strategy:      middle_cycle  |  Routing: expert
  Total params:  10.52M  |  Unique: 3.89M (37.0% of total)
============================================================

Device: cuda  |  Precision: bf16
============================================================

step      0 | loss 10.7821 | lm 10.7710 | aux 0.00043 | z 0.00064 | lr 1.50e-06 | tok/s 24,351
step     10 | loss  9.2143 | lm  9.2031 | aux 0.00038 | z 0.00059 | lr 1.50e-05 | tok/s 51,203
step    100 | loss  7.1203 | lm  7.1089 | aux 0.00024 | z 0.00045 | lr 1.50e-04 | tok/s 52,988
step    500 | loss  5.3102 | lm  5.2990 | aux 0.00008 | z 0.00021 | lr 2.71e-04 | tok/s 53,412

  ── Eval @ step 500 ──
  val_loss=5.2134  val_ppl=184.12
```

---

## Hyperparameter Reference

| Parameter | Default | Description |
|---|---|---|
| `n_recursions` | 3 | Number of recursion steps. More = deeper compute per token |
| `strategy` | `middle_cycle` | Parameter sharing layout. `middle_cycle` is recommended |
| `routing_type` | `expert` | `expert` — model picks tokens. `token` — tokens self-select |
| `alpha` | 0.1 | Router gate scale. Higher = stronger gating effect |
| `aux_loss_w` | 0.001 | Weight for BCE router supervision loss |
| `z_loss_w` | 0.001 | Z-loss weight. Prevents router logit explosion (stability) |
| `cap_warmup` | 1000 | Steps to warm up capacity from 1.0 → target (avoids cold-start) |
| `d_model` | 256 | Hidden embedding dimension |
| `n_heads` | 4 | Number of query attention heads |
| `n_kv_heads` | 4 | Number of KV heads. Set < `n_heads` to enable GQA |
| `d_ff` | auto | FFN inner dim. Defaults to `⌈(8/3 × d_model) / 64⌉ × 64` |
| `max_seq_len` | 256 | Context length |

---

## Project Structure

```
abhinav-mor/
│
├── train.py               # Training entrypoint — python train.py
├── chat.py                # Inference and interactive chat
├── requirements.txt
│
├── configs/
│   └── default.py         # All hyperparameters as Python dataclasses
│                          # Presets: tiny | small | medium
│
├── model/
│   ├── base.py            # LLaMA-style backbone: RMSNorm, RoPE, SwiGLU, GQA
│   ├── sharing.py         # 4 parameter-sharing strategy functions
│   ├── router.py          # Expert-choice + Token-choice routers with loss
│   ├── mor_layer.py       # MoR layer: gather → shared blocks → scatter
│   ├── mor_model.py       # MoRForCausalLM: full model + generation
│   └── kv_cache.py        # Standard + recursive KV caches
│
├── data/
│   └── dataset.py         # tiktoken tokenizer + multi-source dataset loading
│
├── train/
│   ├── trainer.py         # Full training loop: AMP, grad accum, checkpointing
│   └── scheduler.py       # Cosine LR with linear warmup
│
├── eval/
│   └── evaluate.py        # Perplexity evaluation
│
└── scripts/
    ├── train_tiny.sh      # Quick TinyShakespeare run
    └── train_small.sh     # ~85M model training run
```

---

## Citation

If you use this in your work, please cite the original research:

```bibtex
@misc{bae2025mixtureofrecursions,
    title   = {Mixture-of-Recursions: Learning Dynamic Recursive Depths
               for Adaptive Token-Level Computation},
    author  = {Sangmin Bae and Yujin Kim and Reza Bayat and Sungnyun Kim
               and Jiyoun Ha and Tal Schuster and Adam Fisch
               and Hrayr Harutyunyan and Ziwei Ji
               and Aaron Courville and Se-Young Yun},
    year    = {2025},
    eprint  = {2507.10524},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CL},
    url     = {https://arxiv.org/abs/2507.10524}
}
```

---

*Built by Abhinav Singh — implemented using a combination of AI assistance and hand-written code, grounded in the research from KAIST AI, Google DeepMind, and Google Research.*
