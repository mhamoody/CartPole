"""
utils.py — Shared utility functions.
"""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def moving_average(values: list[float], window: int) -> np.ndarray:
    """Compute a simple moving average with the given window size.

    Pads the front with the first value so the output length equals
    the input length (useful for plotting from episode 0).
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    padded = [values[0]] * (window - 1) + list(values)
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def ensure_dir(path: str) -> None:
    """Create directory (and parents) if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)


def load_config(path: str) -> dict:
    """Load a YAML config file and return it as a plain dict."""
    import yaml  # local import keeps the top-level import surface small

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
