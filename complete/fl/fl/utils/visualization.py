"""Visualization utilities for federated learning."""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path


def plot_metrics(
    metrics_history: List[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> None:
    """Plot training metrics over rounds.

    Args:
        metrics_history: List of metric dictionaries per round
        save_path: Optional path to save plot
    """
    if not metrics_history:
        return

    rounds = [m.get("round", i) for i, m in enumerate(metrics_history)]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    if "train_loss" in metrics_history[0]:
        train_losses = [m["train_loss"] for m in metrics_history]
        axes[0].plot(rounds, train_losses, marker="o", label="Train Loss")
    if "eval_loss" in metrics_history[0]:
        eval_losses = [m["eval_loss"] for m in metrics_history]
        axes[0].plot(rounds, eval_losses, marker="s", label="Eval Loss")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over Rounds")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot accuracy
    if "eval_acc" in metrics_history[0]:
        eval_accs = [m["eval_acc"] for m in metrics_history]
        axes[1].plot(rounds, eval_accs, marker="o", color="green")
        axes[1].set_xlabel("Round")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy over Rounds")
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_client_contribution(
    client_samples: Dict[int, int],
    save_path: Optional[str] = None,
) -> None:
    """Plot client data distribution.

    Args:
        client_samples: Dictionary mapping client ID to number of samples
        save_path: Optional path to save plot
    """
    clients = list(client_samples.keys())
    samples = list(client_samples.values())

    plt.figure(figsize=(10, 5))
    plt.bar(clients, samples, color="steelblue", alpha=0.7)
    plt.xlabel("Client ID")
    plt.ylabel("Number of Samples")
    plt.title("Data Distribution Across Clients")
    plt.grid(True, alpha=0.3, axis="y")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_convergence_comparison(
    results: Dict[str, List[float]],
    metric_name: str = "Accuracy",
    save_path: Optional[str] = None,
) -> None:
    """Compare convergence across different methods.

    Args:
        results: Dictionary mapping method name to list of metric values
        metric_name: Name of the metric being plotted
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(10, 6))

    for method_name, values in results.items():
        rounds = list(range(1, len(values) + 1))
        plt.plot(rounds, values, marker="o", label=method_name, linewidth=2)

    plt.xlabel("Round")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Comparison Across Methods")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
