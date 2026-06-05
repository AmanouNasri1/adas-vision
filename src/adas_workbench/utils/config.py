"""Configuration loading and runtime device selection.

The single place that reads YAML config and picks the compute device, so no
other module hard-codes paths, thresholds, or "cuda" vs "cpu". Plain data out:
``load_config`` returns a dict; ``select_device`` returns a string.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict.

    Raises a clear ``FileNotFoundError`` if the file is missing (per the
    "clear error messages, not stack traces" rule in CLAUDE.md §8).
    """
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Config {cfg_path} must be a YAML mapping, got {type(data).__name__}."
        )
    return data


def select_device(prefer: str = "auto") -> str:
    """Pick the compute device: one of "auto", "cuda", "cpu".

    "auto" uses CUDA when available, else CPU. Never raises if CUDA/torch is
    missing — falls back to CPU gracefully (CLAUDE.md §1).
    """
    prefer = (prefer or "auto").lower()
    if prefer == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
