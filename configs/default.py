"""
configs/default.py
──────────────────
All hyperparameters as clean Python dataclasses.
No YAML, no Hydra — just import and go.

Usage:
    from configs.default import ModelConfig, MoRConfig, TrainConfig

    model_cfg = ModelConfig(n_layers=12, d_model=768)
    mor_cfg   = MoRConfig(n_recursions=3, strategy="middle_cycle")
    train_cfg = TrainConfig(max_steps=10_000, lr=3e-4)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional

# ─────────────────────────────────────────────────────────────────
#  Router Config
# ─────────────────────────────────────────────────────────────────
@dataclass
class RouterConfig:
    """Expert-choice router hyperparameters."""

    # Router type: 'linear' (single projection head)
    router_type: Literal["linear"] = "linear"

    # Activation applied to router logits before top-k
    router_func: Literal["sigmoid", "tanh"] = "sigmoid"

    # Scale factor applied to router weights (gating strength)
    alpha: float = 0.1

    # Capacity fraction per recursion step.
    # A list of floats, one per recursion. e.g. [1.0, 0.75, 0.5]
    # If None, auto-computed as [(N-j)/N for j in range(N)]
    capacity: Optional[list] = None

    # Auxiliary loss type for training the router
    #   "aux_loss"   — BCE on the router logits directly
    #   "aux_router" — separate detached router head for BCE
    #   "none"       — no auxiliary loss
    sampling: Literal["aux_loss", "aux_router", "none"] = "aux_loss"

    # Weight for auxiliary BCE loss
    aux_loss_weight: float = 0.001

    # Z-loss weight for router logit regularisation (prevents collapse)
    z_loss_weight: float = 1e-3

    # Whether to use random router (ablation / debugging)
    rand_router: bool = False

    # Temperature scaling for router logits (lower = sharper routing)
    temp: float = 1.0

    # Warmup steps for capacity: starts at 1.0, decays to target
    # Set 0 to disable warmup and use target capacity from step 0
    cap_warmup_steps: int = 1000


# ─────────────────────────────────────────────────────────────────
#  MoR Config
# ─────────────────────────────────────────────────────────────────
@dataclass
class MoRConfig:
    """Mixture-of-Recursions architecture hyperparameters."""

    # Enable MoR routing (if False, model is a plain recursive Transformer)
    enable: bool = True

    # Routing mode: 'expert' (model selects tokens) or 'token' (tokens self-select)
    routing_type: Literal["expert", "token"] = "expert"

    # Number of recursion steps (how many times the shared block is applied)
    n_recursions: int = 3

    # Parameter-sharing strategy: how virtual layers map to shared blocks
    strategy: Literal["cycle", "middle_cycle", "sequence", "middle_sequence"] = "middle_cycle"

    # Gating mode when merging back selected tokens
    gating: Literal["weighted", "identity"] = "weighted"

    # Router settings
    router: RouterConfig = field(default_factory=RouterConfig)

    # KV cache sharing across recursion steps (inference speedup)
    kv_sharing: bool = True


# ─────────────────────────────────────────────────────────────────
#  Model Config
# ─────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """LLaMA-style transformer architecture hyperparameters."""

    # Vocabulary size — set to None to auto-detect from tokenizer
    vocab_size: int = 200_019  # tiktoken o200k_base

    # Maximum sequence / context length
    max_seq_len: int = 512

    # Embedding / hidden dimension
    d_model: int = 512

    # Number of transformer layers (virtual, before parameter sharing)
    n_layers: int = 12

    # Number of query attention heads
    n_heads: int = 8

    # Number of key/value heads (set == n_heads for standard MHA, < for GQA)
    n_kv_heads: int = 8

    # Feed-forward hidden dimension (usually 8/3 * d_model for SwiGLU)
    d_ff: Optional[int] = None  # auto: int(8/3 * d_model), rounded to multiple of 64

    # Dropout rate
    dropout: float = 0.0

    # RMSNorm epsilon
    norm_eps: float = 1e-5

    # RoPE base frequency
    rope_base: float = 10_000.0

    # Weight initialisation std
    init_std: float = 0.02

    # Tie input embedding and output lm_head weights
    tie_weights: bool = True

    def __post_init__(self):
        if self.d_ff is None:
            # SwiGLU standard: (8/3 × d_model) rounded up to nearest multiple of 64
            raw = int(8 / 3 * self.d_model)
            self.d_ff = ((raw + 63) // 64) * 64

        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"


# ─────────────────────────────────────────────────────────────────
#  Training Config
# ─────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    """Training hyperparameters."""

    # Dataset name: "tiny_shakespeare" | "fineweb" | "openwebtext" | path to .txt
    dataset: str = "tiny_shakespeare"

    # Training steps
    max_steps: int = 5_000

    # Warmup steps for learning rate
    warmup_steps: int = 200

    # Batch size (sequences per step)
    batch_size: int = 32

    # Gradient accumulation steps
    grad_accum: int = 1

    # Learning rate
    lr: float = 3e-4

    # AdamW betas
    beta1: float = 0.9
    beta2: float = 0.95

    # Weight decay
    weight_decay: float = 0.1

    # Gradient clipping
    grad_clip: float = 1.0

    # Evaluation interval (steps)
    eval_interval: int = 250

    # Number of batches to average for evaluation
    eval_iters: int = 100

    # Checkpoint save interval (steps)
    save_interval: int = 1000

    # Checkpoint output directory
    out_dir: str = "checkpoints"

    # Mixed precision: "bf16" | "fp16" | "fp32"
    precision: str = "bf16"

    # Compile model with torch.compile (PyTorch 2.0+)
    compile: bool = False

    # Seed for reproducibility
    seed: int = 42

    # Logging interval
    log_interval: int = 10

    # Device override (auto-detected if None)
    device: Optional[str] = None


# ─────────────────────────────────────────────────────────────────
#  Preset Configurations
# ─────────────────────────────────────────────────────────────────

def tiny_config() -> tuple[ModelConfig, MoRConfig, TrainConfig]:
    """
    Tiny model for quick experiments on CPU/single GPU.
    ~10M effective params (with 3x sharing: ~3.5M unique params).
    """
    model = ModelConfig(d_model=256, n_layers=12, n_heads=4, n_kv_heads=4, max_seq_len=256, dropout=0.1)
    mor   = MoRConfig(n_recursions=3, strategy="middle_cycle")
    train = TrainConfig(max_steps=2000, batch_size=16, eval_interval=100, lr=3e-4)
    return model, mor, train


def small_config() -> tuple[ModelConfig, MoRConfig, TrainConfig]:
    """
    Small model (~85M effective, ~28M unique params).
    """
    model = ModelConfig(d_model=512, n_layers=12, n_heads=8, n_kv_heads=4, max_seq_len=512, dropout=0.0)
    mor   = MoRConfig(n_recursions=3, strategy="middle_cycle")
    train = TrainConfig(max_steps=20_000, batch_size=32, grad_accum=4, lr=2e-4, warmup_steps=500, compile=True)
    return model, mor, train


def medium_config() -> tuple[ModelConfig, MoRConfig, TrainConfig]:
    """
    Medium model (~360M effective, ~120M unique params).
    """
    model = ModelConfig(d_model=1024, n_layers=24, n_heads=16, n_kv_heads=8, max_seq_len=2048, dropout=0.0)
    mor   = MoRConfig(n_recursions=3, strategy="middle_cycle")
    train = TrainConfig(max_steps=100_000, batch_size=64, grad_accum=8, lr=1e-4, warmup_steps=2000, precision="bf16", compile=True)
    return model, mor, train


def large_config() -> tuple[ModelConfig, MoRConfig, TrainConfig]:
    """
    ~1B effective parameter model with massive 128k context window.
    WARNING: This will immediately OOM on a standard Kaggle GPU or laptop!
    """
    model = ModelConfig(
        d_model=2048, 
        n_layers=36,       # 36 virtual layers
        n_heads=16, 
        n_kv_heads=8,      # GQA to save some memory
        max_seq_len=128000,# Massive context window
        dropout=0.0
    )
    mor   = MoRConfig(n_recursions=3, strategy="middle_cycle")
    # Batch size forced to 1 because of the massive context window
    train = TrainConfig(max_steps=200_000, batch_size=1, grad_accum=128, lr=1e-4, warmup_steps=2000, precision="bf16", compile=True)
    return model, mor, train


PRESETS = {
    "tiny":     tiny_config,
    "small":    small_config,
    "medium":   medium_config,
    "large":    large_config,
}
