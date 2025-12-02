"""Metrics computation utilities."""

import torch
from typing import Tuple


def compute_accuracy(
    outputs: torch.Tensor, targets: torch.Tensor
) -> float:
    """Compute classification accuracy.

    Args:
        outputs: Model predictions (logits or probabilities)
        targets: Ground truth labels

    Returns:
        Accuracy as a float between 0 and 1
    """
    predictions = outputs.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0


def compute_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    criterion: torch.nn.Module = None,
) -> float:
    """Compute loss.

    Args:
        outputs: Model predictions
        targets: Ground truth labels
        criterion: Loss function (defaults to CrossEntropyLoss)

    Returns:
        Loss value
    """
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    loss = criterion(outputs, targets)
    return loss.item()


def compute_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Compute accuracy and loss on a dataset.

    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device to run computations on

    Returns:
        Tuple of (accuracy, loss)
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return accuracy, avg_loss
