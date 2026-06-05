"""Configuration loading, runtime device selection, and output-path resolution.

The single place that reads YAML config, picks the compute device, and decides
where outputs go — so no other module hard-codes paths, thresholds, or
"cuda" vs "cpu". Plain data out: dicts, strings, and Paths.
"""
from __future__ import annotations

import os
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


def _external_available(base: str | Path) -> bool:
    """True if the drive root of ``base`` exists (e.g. ``E:\\`` is mounted)."""
    try:
        anchor = Path(base).anchor  # e.g. "E:\\"
        return bool(anchor) and os.path.exists(anchor)
    except Exception:
        return False


def resolve_output_dir(paths_config: dict[str, Any] | None, kind: str = "videos") -> Path:
    """Return a writable output directory for ``kind`` ("videos" or "logs").

    Prefers the external HDD base (e.g. ``D:\\adas_outputs``) from
    ``configs/paths.yaml`` when that drive is mounted; otherwise falls back to
    the in-repo ``outputs/`` folder so the pipeline never crashes when E: is
    absent (mirrors the CUDA->CPU fallback). The chosen directory is created.
    """
    outputs = (paths_config or {}).get("outputs", {})
    external_base = outputs.get("external_base", "D:/adas_outputs")
    local_base = outputs.get("local_base", "outputs")
    subdir = outputs.get(f"{kind}_subdir", kind)

    if _external_available(external_base):
        target = Path(external_base) / subdir
    else:
        target = Path(local_base) / subdir
    target.mkdir(parents=True, exist_ok=True)
    return target
