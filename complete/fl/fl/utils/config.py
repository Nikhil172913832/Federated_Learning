"""Configuration management utilities."""

import yaml
import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file.

    Args:
        config_path: Path to config file. If None, loads default config.

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default config path
        config_path = Path(__file__).parents[2] / "config" / "default.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix in [".yaml", ".yml"]:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == ".json":
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    return config or {}


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary.

    Args:
        config: Configuration to validate

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_keys = ["train", "data"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    # Validate train config
    train_config = config["train"]
    if "num_server_rounds" not in train_config:
        raise ValueError("Missing 'num_server_rounds' in train config")
    if "local_epochs" not in train_config:
        raise ValueError("Missing 'local_epochs' in train config")

    # Validate data config
    data_config = config["data"]
    if "dataset" not in data_config:
        raise ValueError("Missing 'dataset' in data config")

    return True
