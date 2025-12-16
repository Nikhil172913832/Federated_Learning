"""Dataset utilities for federated learning.

This module provides data loading and partitioning for FL experiments.
"""

from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
import numpy as np


def load_mnist_federated(
    num_clients: int = 10,
    batch_size: int = 32,
    iid: bool = True,
    alpha: float = 0.5,
) -> Tuple[List[DataLoader], DataLoader]:
    """Load MNIST dataset partitioned for federated learning.
    
    Args:
        num_clients: Number of clients to partition data for
        batch_size: Batch size for data loaders
        iid: Whether to use IID partitioning
        alpha: Dirichlet alpha for non-IID partitioning (lower = more skew)
        
    Returns:
        Tuple of (list of client train loaders, global test loader)
    """
    # Load MNIST from HuggingFace
    dataset = load_dataset("ylecun/mnist")
    
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    # Partition training data
    if iid:
        client_loaders = _partition_iid(train_data, num_clients, batch_size)
    else:
        client_loaders = _partition_noniid(train_data, num_clients, batch_size, alpha)
    
    # Create global test loader
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_fn,
    )
    
    return client_loaders, test_loader


def _partition_iid(
    dataset,
    num_clients: int,
    batch_size: int,
) -> List[DataLoader]:
    """Partition dataset in IID manner.
    
    Args:
        dataset: HuggingFace dataset
        num_clients: Number of clients
        batch_size: Batch size
        
    Returns:
        List of DataLoaders for each client
    """
    total_size = len(dataset)
    indices = np.random.permutation(total_size)
    client_size = total_size // num_clients
    
    client_loaders = []
    for i in range(num_clients):
        start_idx = i * client_size
        end_idx = start_idx + client_size if i < num_clients - 1 else total_size
        client_indices = indices[start_idx:end_idx]
        
        client_dataset = dataset.select(client_indices.tolist())
        loader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_fn,
        )
        client_loaders.append(loader)
    
    return client_loaders


def _partition_noniid(
    dataset,
    num_clients: int,
    batch_size: int,
    alpha: float,
) -> List[DataLoader]:
    """Partition dataset in non-IID manner using Dirichlet distribution.
    
    Args:
        dataset: HuggingFace dataset
        num_clients: Number of clients
        batch_size: Batch size
        alpha: Dirichlet concentration parameter
        
    Returns:
        List of DataLoaders for each client
    """
    # Get labels
    labels = np.array([item["label"] for item in dataset])
    num_classes = len(np.unique(labels))
    
    # Dirichlet partitioning
    client_indices = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        # Sample from Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        # Split indices according to proportions
        splits = np.split(idx_k, proportions)
        for i, split in enumerate(splits):
            client_indices[i].extend(split.tolist())
    
    # Create DataLoaders
    client_loaders = []
    for indices in client_indices:
        if len(indices) == 0:
            continue
        client_dataset = dataset.select(indices)
        loader = DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_collate_fn,
        )
        client_loaders.append(loader)
    
    return client_loaders


def _collate_fn(batch):
    """Collate function to convert HuggingFace batch to tensors.
    
    Args:
        batch: List of examples from dataset
        
    Returns:
        Dictionary with image and label tensors
    """
    images = torch.stack([
        torch.FloatTensor(item["image"]).unsqueeze(0) / 255.0
        for item in batch
    ])
    labels = torch.LongTensor([item["label"] for item in batch])
    
    # Normalize
    images = (images - 0.5) / 0.5
    
    return {"image": images, "label": labels}
