"""
data/dataset.py
───────────────
Multi-format dataset loading formor.

Supported sources:
  • "tiny_shakespeare" — downloads from karpathy/char-rnn (classic)
  • "fineweb"          — HuggingFace FineWeb-Edu (large scale)
  • "openwebtext"      — HuggingFace OpenWebText
  • <path>             — local .txt file

Tokenizer: tiktoken o200k_base (same as GPT-4o, 200k vocab)
"""
from __future__ import annotations
import os
import urllib.request
from typing import Optional, Tuple

import torch


# ─────────────────────────────────────────────────────────────────
#  Tokenizer (global, lazy-loaded)
# ─────────────────────────────────────────────────────────────────

_enc = None

def get_tokenizer():
    global _enc
    if _enc is None:
        import tiktoken
        _enc = tiktoken.get_encoding("o200k_base")
    return _enc

def encode(text: str) -> list:
    return get_tokenizer().encode(text)

def decode(ids: list) -> str:
    return get_tokenizer().decode(ids)


# ─────────────────────────────────────────────────────────────────
#  Dataset Loading
# ─────────────────────────────────────────────────────────────────

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def _load_tiny_shakespeare(cache_dir: str = ".cache") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "tinyshakespeare.txt")
    if not os.path.exists(path):
        print(f"Downloading TinyShakespeare → {path}")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
        print("Done.")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_local_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_hf_dataset(name: str, split: str = "train", max_samples: Optional[int] = None) -> str:
    """Load a HuggingFace text dataset and concatenate into a single string."""
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("Install HuggingFace datasets: pip install datasets")

    HF_NAMES = {
        "fineweb":     ("HuggingFaceFW/fineweb-edu", "default"),
        "openwebtext": ("openwebtext", None),
    }

    if name not in HF_NAMES:
        raise ValueError(f"Unknown HF dataset '{name}'. Valid: {list(HF_NAMES)}")

    ds_id, ds_config = HF_NAMES[name]
    print(f"Loading {name} from HuggingFace...")

    if ds_config:
        ds = hf_load(ds_id, ds_config, split=split, streaming=False)
    else:
        ds = hf_load(ds_id, split=split, streaming=False)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    text_col = "text"
    print(f"Concatenating {len(ds)} samples...")
    return "\n\n".join(str(row[text_col]) for row in ds)


def get_dataset(
    source: str,
    train_split: float = 0.9,
    max_samples: Optional[int] = None,
    cache_dir: str = ".cache",
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load, tokenize, and split a dataset.

    Args:
        source:      "tiny_shakespeare" | "fineweb" | "openwebtext" | /path/to/file.txt
        train_split: fraction of data for training (rest is validation)
        max_samples: limit number of samples (for HF datasets)
        cache_dir:   local cache directory
        device:      torch device for tensors

    Returns:
        (train_data, val_data) — 1D LongTensors of token IDs
    """
    # ── Load raw text ──
    if source == "tiny_shakespeare":
        text = _load_tiny_shakespeare(cache_dir)
    elif os.path.isfile(source):
        text = _load_local_txt(source)
    else:
        text = _load_hf_dataset(source, max_samples=max_samples)

    print(f"Dataset: {len(text):,} characters")

    # ── Tokenise ──
    enc = get_tokenizer()
    print(f"Tokenising with tiktoken {enc.name} (vocab={enc.n_vocab:,})...")
    ids = enc.encode(text)
    data = torch.tensor(ids, dtype=torch.long)

    print(f"Tokenised: {len(data):,} tokens")

    # ── Train / val split ──
    n = int(train_split * len(data))
    train_data = data[:n].to(device)
    val_data   = data[n:].to(device)

    print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")
    return train_data, val_data


# ─────────────────────────────────────────────────────────────────
#  Batch Sampler
# ─────────────────────────────────────────────────────────────────

def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random batch of (input, target) pairs.

    Args:
        data:       1D tensor of token IDs
        block_size: context length
        batch_size: number of sequences per batch
        device:     move tensors to this device

    Returns:
        x: (batch_size, block_size) input tokens
        y: (batch_size, block_size) target tokens (shifted by 1)
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i:i + block_size    ] for i in ix])
    y  = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    if device is not None:
        x, y = x.to(device), y.to(device)
    return x, y
