"""Data loading utilities."""

from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner


def apply_transforms(batch, transforms):
    """Apply transforms to a batch.

    Args:
        batch: Batch from dataset
        transforms: Torchvision transforms

    Returns:
        Transformed batch
    """
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch


def load_partition_data(
    partition_id: int,
    num_partitions: int,
    dataset: str = "albertvillanova/medmnist-v2",
    subset: str = "pneumoniamnist",
    batch_size: int = 32,
    partitioner = None,
) -> Tuple[DataLoader, DataLoader]:
    """Load federated data partition.

    Args:
        partition_id: ID of the partition to load
        num_partitions: Total number of partitions
        dataset: HuggingFace dataset name
        subset: Dataset subset
        batch_size: Batch size for dataloaders
        partitioner: Custom partitioner (defaults to IID)

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Setup transforms
    pytorch_transforms = Compose([
        Resize(size=(28, 28)),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5]),
    ])

    # Load federated dataset
    if partitioner is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)

    fds = FederatedDataset(
        dataset=dataset,
        subset=subset,
        trust_remote_code=True,
        partitioners={"train": partitioner},
    )

    # Load specific partition
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Apply transforms
    partition_train_test = partition_train_test.with_transform(
        lambda batch: apply_transforms(batch, pytorch_transforms)
    )

    # Create dataloaders
    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=batch_size,
        shuffle=True,
    )
    testloader = DataLoader(
        partition_train_test["test"],
        batch_size=batch_size,
    )

    return trainloader, testloader
