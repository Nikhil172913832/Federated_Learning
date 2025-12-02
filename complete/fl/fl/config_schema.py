"""
Configuration Schema and Validation for Federated Learning

This file defines the expected structure and validation for FL configs.
"""

from typing import Dict, Any, Optional
import yaml


class ConfigSchema:
    """Schema definition for federated learning configuration."""

    REQUIRED_FIELDS = {
        "experiment": ["name", "seed"],
        "model": ["architecture", "num_classes"],
        "federated": ["num_clients", "clients_per_round", "num_rounds"],
        "train": ["local_epochs", "lr", "batch_size"],
        "data": ["dataset"],
    }

    OPTIONAL_FIELDS = {
        "experiment": ["description", "tags"],
        "model": ["input_channels", "hidden_dims"],
        "federated": ["aggregation_strategy", "client_selection"],
        "train": ["optimizer", "scheduler"],
        "data": ["subset", "non_iid", "augmentation"],
        "privacy": ["enable_dp", "epsilon", "delta"],
        "security": ["secure_agg", "byzantine_robust"],
        "tracking": ["use_mlflow", "use_wandb", "wandb_project"],
    }

    @classmethod
    def validate(cls, config: Dict[str, Any]) -> bool:
        """Validate configuration dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            True if valid

        Raises:
            ValueError: If configuration is invalid
        """
        # Check required top-level sections
        for section, fields in cls.REQUIRED_FIELDS.items():
            if section not in config:
                raise ValueError(f"Missing required section: {section}")

            section_config = config[section]
            for field in fields:
                if field not in section_config:
                    raise ValueError(f"Missing required field: {section}.{field}")

        return True

    @classmethod
    def load_and_validate(cls, config_path: str) -> Dict[str, Any]:
        """Load and validate configuration file.

        Args:
            config_path: Path to YAML config file

        Returns:
            Validated configuration dictionary
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        cls.validate(config)
        return config

    @classmethod
    def get_template(cls) -> Dict[str, Any]:
        """Get configuration template with all fields.

        Returns:
            Template configuration dictionary
        """
        return {
            "experiment": {
                "name": "federated_experiment",
                "description": "Federated learning experiment",
                "seed": 42,
                "tags": ["baseline", "fedavg"],
            },
            "model": {
                "architecture": "cnn",
                "input_channels": 1,
                "num_classes": 10,
            },
            "federated": {
                "num_clients": 100,
                "clients_per_round": 10,
                "num_rounds": 100,
                "aggregation_strategy": "fedavg",
                "client_selection": "random",
            },
            "train": {
                "local_epochs": 5,
                "lr": 0.001,
                "batch_size": 32,
                "optimizer": "adam",
            },
            "data": {
                "dataset": "mnist",
                "non_iid": {
                    "enabled": False,
                    "type": "label_skew",
                    "alpha": 0.5,
                },
            },
            "privacy": {
                "enable_dp": False,
                "epsilon": 1.0,
                "delta": 1e-5,
                "max_grad_norm": 1.0,
            },
            "security": {
                "secure_agg": False,
                "byzantine_robust": False,
                "strategy": "krum",
            },
            "tracking": {
                "use_mlflow": True,
                "use_wandb": False,
                "wandb_project": "federated-learning",
            },
        }


def save_template(output_path: str = "config_template.yaml") -> None:
    """Save configuration template to file.

    Args:
        output_path: Path to save template
    """
    template = ConfigSchema.get_template()
    with open(output_path, "w") as f:
        yaml.dump(template, f, default_flow_style=False, sort_keys=False)
    print(f"Template saved to {output_path}")


if __name__ == "__main__":
    save_template()
