"""Config loading and reproducible seeding utilities for the Flower app."""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # Will raise at runtime if YAML is actually used


def _default_config_path() -> Path:
    # Resolve to repo-local default at `complete/fl/config/default.yaml`
    # This file lives in `complete/fl/fl/config.py` => ascend two levels
    app_root = Path(__file__).resolve().parents[1]
    return app_root / "config" / "default.yaml"


def load_run_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load run configuration from YAML or JSON.

    Precedence:
    - explicit `config_path` if provided
    - env var `FL_CONFIG_PATH`
    - repo default `config/default.yaml`
    """

    cfg_path_str = (
        config_path
        or os.environ.get("FL_CONFIG_PATH")
        or str(_default_config_path())
    )
    cfg_path = Path(cfg_path_str)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError(
                "PyYAML not installed but a YAML config was requested. Install pyyaml."
            )
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif cfg_path.suffix.lower() == ".json":
        with cfg_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(
            f"Unsupported config format: {cfg_path.suffix}. Use .yaml/.yml or .json."
        )

    return data or {}


def set_global_seeds(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior where possible
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def merge_with_context_defaults(context_cfg: Dict[str, Any], file_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge file config into context config, file config takes precedence.

    Converts common keys to expected context names when present.
    """

    merged = dict(context_cfg or {})
    # Map common aliases
    aliases = {
        "num_server_rounds": "num-server-rounds",
        "num-server-rounds": "num-server-rounds",
        "fraction": "fraction-train",
        "fraction_train": "fraction-train",
        "fraction-train": "fraction-train",
        "local_epochs": "local-epochs",
        "local-epochs": "local-epochs",
        "lr": "lr",
        "seed": "seed",
    }
    for key, value in (file_cfg or {}).items():
        mapped = aliases.get(key, key)
        merged[mapped] = value
    return merged


