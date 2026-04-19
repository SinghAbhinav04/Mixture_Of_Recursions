# MoR
### *Mixture-of-Recursions Language Model — Production Implementation*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
> 📄 [Read the research paper](https://arxiv.org/abs/2507.10524)

---

## Why MoR? (MoR vs Standard GPT)

Standard transformers (GPT, LLaMA, etc.) waste compute by applying **the same depth of computation to every token**, whether it's a punctuation mark or a complex reasoning step.

**MoR fixes this with adaptive computation:**

```
Standard GPT (12 layers, wasteful):

  "The"    → [L1] → [L2] → [L3] → [L4] → ... → [L12] → output    (12 layers, easy word)
  "cat"    → [L1] → [L2] → [L3] → [L4] → ... → [L12] → output    (12 layers, easy word)
  "proves" → [L1] → [L2] → [L3] → [L4] → ... → [L12] → output    (12 layers, hard reasoning)
  ──────────────────────────────────────────────────────────────────
  Every token pays full cost. No savings for easy tokens.


MoR (12 virtual layers with 3× recursion, smart):

  "The"    → [Shared Blocks × 1 recursion] ──────→ EXIT EARLY     (4 layers, saves 67%)
  "cat"    → [Shared Blocks × 2 recursions] ────→ EXIT             (8 layers, saves 33%)
  "proves" → [Shared Blocks × 3 recursions] → FULL DEPTH          (12 layers, full compute)
  ──────────────────────────────────────────────────────────────────
  Easy tokens exit early. Hard tokens get full depth.
  Same weights reused across recursions → 3× fewer unique params!
```

### Key Advantages Over GPT

| Property | Standard GPT | MoR (Ours) |
|----------|-------------|------------|
| **Compute per token** | Fixed (all layers) | Adaptive (router decides depth) |
| **Parameters** | All unique layers | Shared blocks → **~3× fewer unique params** |
| **Inference speed** | O(L × T²) per step | Up to **2× faster** (early exit) |
| **Context window** | Limited by memory | Up to **128K tokens** (SDPA + no O(N²) masks) |
| **FLOPs at same quality** | Baseline | **~40-60% less** (paper Table 1) |
| **Architecture** | Static depth | **Dynamic computation** — adapts per token |

### Context Window

Our context window depends on the model preset:

| Preset | Context Window | Use Case |
|--------|---------------|----------|
| `tiny` | **256 tokens** | Quick experiments, debugging |
| `small` | **512 tokens** | Small-scale training |
| `medium` | **2,048 tokens** | Production training |
| `large` | **128,000 tokens** | Long-document, enterprise-scale |

The 128K context window is enabled by **Flash Attention via PyTorch's SDPA** — we never materialize the O(N²) causal mask. Instead, `F.scaled_dot_product_attention(is_causal=True)` handles masking internally with O(N) memory.

---

## Research Foundation

> **Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation**
> Sangmin Bae, Yujin Kim, Reza Bayat, Sungnyun Kim, Jiyoun Ha, Tal Schuster, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Aaron Courville, Se-Young Yun
> *NeurIPS 2025 · KAIST AI · Google DeepMind · Google Research · Mila*
> [arXiv:2507.10524](https://arxiv.org/abs/2507.10524)

Our implementation is **aligned with Google's official reference repo** and implements all key mechanisms from the paper:

- **Expert-choice routing** (§2.2) — model selects top-k tokens per recursion step
- **Token-choice routing** (§2.2.1) — one-shot softmax assignment over N_r experts
- **4 parameter sharing strategies** (§2.2.2) — cycle, middle_cycle, sequence, middle_sequence
- **Gated residual merging** (Eq 2.1) — `x + gate × block(selected_x)`
- **BCE auxiliary loss + z-loss** (§4.2) — router training stability
- **Loss-free balancing** — learnable router bias (no gradient-based balancing loss)
- **Balancing warmup** — forces all tokens to max depth during early training
- **Linear / MLP / Wide-MLP routers** — configurable router architectures
- **Cosine capacity warmup** — prevents cold-start routing collapse
- **Recursion-wise KV caching** — solves the missing KV cache problem at inference

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
│       ├── RMSNorm → CausalSelfAttention (RoPE + GQA + SDPA Flash Attention)
│       └── RMSNorm → SwiGLUFFN
│
├── [MoR Recursion Loop]                  ← N_r steps, reusing shared blocks
│   ├── Step 0: MoRLayer (capacity 100%)
│   │   ├── Router → scores → top-k selection → SORT (causal order)
│   │   ├── gather(x, top_k_idx)          ← physical sequence reduction
│   │   ├── run shared blocks on reduced  ← real compute savings
│   │   └── scatter_add(x, gate × output) ← gated residual merge
│   ├── Step 1: MoRLayer (capacity ~67%)
│   └── Step 2: MoRLayer (capacity ~33%)
│
├── [Output Head]
│   ├── RMSNorm (final)
│   └── lm_head: Linear (weights tied with tok_emb)
│
└── [Training Losses]
    ├── lm_loss:   cross-entropy next-token prediction
    ├── aux_loss:  BCE router supervision / balancing loss
    └── z_loss:    log-sum-exp penalty (router stability)
```

### Backbone: LLaMA-Style Decoder

Every `TransformerBlock` implements the modern LLM recipe:

| Component | Design Choice | Why |
|---|---|---|
| **Norm** | RMSNorm | Cheaper than LayerNorm, no mean subtraction |
| **Position** | RoPE (Rotary Embeddings) | Generalises to longer sequences, no learned positions |
| **FFN** | SwiGLU (`SiLU(W_gate·x) ⊙ W_up·x → W_down`) | Higher quality than GELU at same params |
| **Attention** | GQA + SDPA (Flash Attention) | O(N) memory, 128K context, faster inference |
| **Bias** | None in linear layers | Follows LLaMA/Gemma best practices |
| **Tying** | `lm_head.weight = tok_emb.weight` | Reduces params, improves training |

### Routing Modes

#### Expert-Choice (Recommended)

The model decides which tokens need deep processing. Each recursion step independently selects top-k tokens by router score.

```python
logits     = Router(x)                           # (B, T, 1)
gate       = sigmoid(logits) * alpha             # gating weights
selected   = topk(gate, k=capacity * T)          # select + SORT for causal order
x_reduced  = gather(x, selected)                 # physical reduction
output     = TransformerBlocks(x_reduced)         # process only k tokens
x          = scatter_add(x, selected, output * gate)  # merge back
```

#### Token-Choice (Paper §2.2.1)

One-shot assignment: each token commits to a recursion depth at the start.

```python
logits       = Router(x)                         # (B, T, N_r)  — scores over N_r experts
assignments  = softmax(logits).argmax(dim=-1)    # each token → one depth
# At recursion r: only tokens with assignment >= r participate
```

Supports both **gradient-based balancing** (`Σ P_i × f_i`) and **loss-free balancing** (learnable bias).

### Router Types

| Type | Architecture | Best For |
|------|-------------|----------|
| `linear` | `Linear(d → out)` | Expert-choice (default, fastest) |
| `mlp` | `Linear(d → 2d) → GELU → Linear(2d → out)` | Token-choice (better quality) |
| `wide_mlp` | Same as mlp | Config clarity |

### Parameter Sharing Strategies

`n_layers` virtual transformer layers are mapped to `n_unique` shared blocks (example: L=12, N_r=3):

| Strategy | Virtual → Block mapping | Unique blocks |
|---|---|---|
| `cycle` | `[0,1,2,3, 0,1,2,3, 0,1,2,3]` | 4 |
| `middle_cycle` | `[0, 1,2,3,1,2,3,1,2,3, 4]` | 5 ← **recommended** |
| `sequence` | `[0,0,0,0, 1,1,1,1, 2,2,2,2]` | 4 |
| `middle_sequence` | `[0, 1,1,1,2,2,2,3,3,3, 4]` | 5 |

`middle_cycle` gives unique first/last blocks and cycles the middle — best empirical balance.

### KV Cache at Inference

At inference, tokens that exit early still need their KV entries available for future token generation. `RecursiveKVCache` stores KV pairs per `(block_idx, recursion_step)` and carries forward entries — **solving the missing KV problem** described in §3 of the paper.

---

## Installation

```bash
# Clone and enter the project
git clone https://github.com/SinghAbhinav04/Mixture_Of_Recursions.git abhinav-mor
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
# Tiny  — ~3.5M unique params, 256 ctx, runs on CPU
python train.py --preset tiny --max_steps 2000

# Small — ~28M unique params, 512 ctx, needs GPU
python train.py --preset small --max_steps 20000

# Medium — ~120M unique params, 2K ctx, A100/H100 recommended
python train.py --preset medium --dataset fineweb

# Large — ~1B unique params, 128K ctx (Requires massive VRAM!)
python train.py --preset large --dataset openwebtext
```

### Full manual control

```bash
python train.py \
    --dataset tiny_shakespeare \
    --d_model 512 --n_layers 12 --n_heads 8 --n_kv_heads 4 \
    --n_recursions 3 --strategy middle_cycle \
    --routing_type expert --router_type linear --alpha 0.1 \
    --aux_loss_w 0.001 --z_loss_w 0.001 \
    --max_steps 10000 --batch_size 32 --grad_accum 4 \
    --lr 3e-4 --warmup_steps 500 \
    --precision bf16 \
    --out_dir checkpoints/my_run
```

### Token-choice with MLP router and loss-free balancing

```bash
python train.py \
    --routing_type token --router_type mlp \
    --balancing loss_free --bal_warmup_steps 500 \
    --n_recursions 3 --strategy middle_cycle
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
Strategy='middle_cycle'  L=12  N_r=3  unique_blocks=5
Routing: [0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 4]
[MoRForCausalLM] capacities=[1.0, 0.667, 0.333]

============================================================
  MoR
  d_model:       256  |  n_layers: 12 virtual (5 unique, 3×)
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
| `n_recursions` | 3 | Number of recursion steps per MoR layer |
| `strategy` | `middle_cycle` | Parameter sharing layout. `middle_cycle` is recommended |
| `routing_type` | `expert` | `expert` — model picks tokens. `token` — tokens self-select |
| `router_type` | `linear` | Router head: `linear`, `mlp`, or `wide_mlp` |
| `alpha` | 0.1 | Router gate scale. Higher = stronger gating effect |
| `aux_loss_w` | 0.001 | Weight for BCE/balancing loss |
| `z_loss_w` | 0.001 | Z-loss weight (prevents router logit explosion) |
| `balancing` | `loss` | Token-choice balancing: `loss` (gradient) or `loss_free` (bias) |
| `bal_warmup_steps` | 0 | Steps to force all tokens to max depth (token-choice) |
| `cap_warmup` | 1000 | Steps for capacity warmup from 1.0 → target |
| `d_model` | 256 | Hidden embedding dimension |
| `n_heads` | 4 | Number of query attention heads |
| `n_kv_heads` | 4 | Number of KV heads. Set < `n_heads` to enable GQA |
| `d_ff` | auto | FFN inner dim. Defaults to `⌈(8/3 × d_model) / 64⌉ × 64` |
| `max_seq_len` | 256 | Context window (tokens) |

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
│                          # Presets: tiny | small | medium | large
│
├── model/
│   ├── base.py            # LLaMA-style backbone: RMSNorm, RoPE, SwiGLU, GQA, SDPA
│   ├── sharing.py         # 4 parameter-sharing strategy functions
│   ├── router.py          # Expert-choice + Token-choice routers + MLP/Wide-MLP heads
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

*Built by Abhinav Singh — implemented using a combination of AI assistance and hand-written code, aligned with the official Google reference implementation. Grounded in research from KAIST AI, Google DeepMind, and Google Research.*
