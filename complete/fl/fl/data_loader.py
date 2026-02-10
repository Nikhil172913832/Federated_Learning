"""Thread-safe data loading for federated learning."""

import threading
from typing import Tuple, Optional, Dict, Any
import torch
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter

from fl.partitioning import build_partitioner


class FederatedDataLoader:
    """Thread-safe federated data loader.

    Encapsulates dataset loading and partitioning logic to avoid global state.
    """

    _lock = threading.Lock()
    _instances: Dict[str, "FederatedDataLoader"] = {}

    def __init__(self, config: Dict[str, Any]):
        """Initialize data loader with configuration.

        Args:
            config: Configuration dictionary containing data, preprocess, and augmentation settings
        """
        self.config = config
        self._fds: Optional[FederatedDataset] = None
        self._transforms = self._build_transforms()

    @classmethod
    def get_instance(cls, config: Dict[str, Any]) -> "FederatedDataLoader":
        """Get or create singleton instance for given config.

        Args:
            config: Configuration dictionary

        Returns:
            FederatedDataLoader instance
        """
        config_key = str(sorted(config.items()))

        with cls._lock:
            if config_key not in cls._instances:
                cls._instances[config_key] = cls(config)
            return cls._instances[config_key]

    def _build_transforms(self) -> Compose:
        """Build image transforms from config."""
        preprocess_cfg = self.config.get("preprocess", {})
        aug_cfg = self.config.get("augmentation", {})

        transforms = []

        if preprocess_cfg.get("resize"):
            transforms.append(Resize(preprocess_cfg["resize"]))

        if aug_cfg.get("enabled", False):
            params = aug_cfg.get("params", {})
            if params.get("hflip", True):
                transforms.append(RandomHorizontalFlip(p=0.5))
            if params.get("rotation_degrees", 0) > 0:
                transforms.append(RandomRotation(degrees=params["rotation_degrees"]))
            if params.get("color_jitter", False):
                transforms.append(
                    ColorJitter(
                        brightness=params.get("brightness", 0.0),
                        contrast=params.get("contrast", 0.0),
                        saturation=params.get("saturation", 0.0),
                        hue=params.get("hue", 0.0),
                    )
                )

        transforms.extend(
            [
                ToTensor(),
                Normalize(
                    mean=preprocess_cfg.get("normalize_mean", [0.5]),
                    std=preprocess_cfg.get("normalize_std", [0.5]),
                ),
            ]
        )

        return Compose(transforms)

    def _apply_transforms(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transforms to batch."""
        batch["image"] = [self._transforms(img) for img in batch["image"]]
        return batch

    def _init_federated_dataset(self, num_partitions: int):
        """Initialize federated dataset if not already done."""
        if self._fds is not None:
            return

        with self._lock:
            if self._fds is not None:
                return

            data_cfg = self.config.get("data", {})
            dataset = data_cfg.get("dataset", "albertvillanova/medmnist-v2")
            subset = data_cfg.get("subset", "pneumoniamnist")
            non_iid_cfg = data_cfg.get("non_iid")

            partitioner = build_partitioner(
                num_partitions=num_partitions, cfg=non_iid_cfg
            )

            self._fds = FederatedDataset(
                dataset=dataset,
                subset=subset,
                trust_remote_code=True,
                partitioners={"train": partitioner},
            )

    def load_partition(
        self, partition_id: int, num_partitions: int
    ) -> Tuple[DataLoader, DataLoader]:
        """Load data partition for a client.

        Args:
            partition_id: Partition ID (0 to num_partitions-1)
            num_partitions: Total number of partitions

        Returns:
            Tuple of (train_loader, test_loader)

        Raises:
            ValueError: If partition_id or num_partitions are invalid
        """
        if num_partitions <= 0:
            raise ValueError(f"num_partitions must be positive, got {num_partitions}")
        if not (0 <= partition_id < num_partitions):
            raise ValueError(
                f"partition_id must be in [0, {num_partitions}), got {partition_id}"
            )

        self._init_federated_dataset(num_partitions)

        partition = self._fds.load_partition(partition_id)
        partition_split = partition.train_test_split(test_size=0.2, seed=42)
        partition_split = partition_split.with_transform(self._apply_transforms)

        batch_size = int(self.config.get("data", {}).get("batch_size", 32))

        train_loader = DataLoader(
            partition_split["train"], batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(partition_split["test"], batch_size=batch_size)

        return train_loader, test_loader
