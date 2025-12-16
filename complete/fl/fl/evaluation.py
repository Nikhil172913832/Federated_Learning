"""Comprehensive model evaluation suite.

This module provides tools for multi-metric evaluation including:
- Precision, recall, F1 scores per class
- Confusion matrix visualization
- Calibration error analysis
- Per-client fairness evaluation
"""

import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """Comprehensive model evaluation with multiple metrics."""

    def __init__(self, num_classes: int = 10, class_names: Optional[List[str]] = None):
        """Initialize evaluator.
        
        Args:
            num_classes: Number of classes
            class_names: Optional list of class names for visualization
        """
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]

    def evaluate_global_model(
        self,
        model: nn.Module,
        test_loader,
        device: torch.device,
        save_visualizations: bool = True,
    ) -> Dict[str, float]:
        """Comprehensive evaluation of global model.
        
        Args:
            model: Neural network model
            test_loader: DataLoader for test data
            device: Device to run evaluation on
            save_visualizations: Whether to save confusion matrix plots
            
        Returns:
            Dictionary with comprehensive metrics
        """
        model.to(device)
        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probs = F.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Compute comprehensive metrics
        metrics = self._compute_metrics(all_labels, all_preds, all_probs)
        metrics["test_loss"] = total_loss / len(test_loader)

        # Generate visualizations
        if save_visualizations and MATPLOTLIB_AVAILABLE:
            cm = confusion_matrix(all_labels, all_preds)
            fig = self._plot_confusion_matrix(cm)
            
            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_figure(fig, "confusion_matrix.png")
                except Exception as e:
                    logger.warning(f"Could not log confusion matrix to MLflow: {e}")
            
            plt.close(fig)

        return metrics

    def _compute_metrics(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        probs: np.ndarray,
    ) -> Dict[str, float]:
        """Compute comprehensive metrics.
        
        Args:
            labels: Ground truth labels
            preds: Predicted labels
            probs: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        # Overall accuracy
        accuracy = (preds == labels).mean()

        # Per-class precision, recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )

        # Macro averages
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()

        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )

        # Calibration error
        calibration_error = self._compute_calibration_error(probs, labels)

        metrics = {
            "accuracy": float(accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_precision),
            "weighted_recall": float(weighted_recall),
            "weighted_f1": float(weighted_f1),
            "calibration_error": float(calibration_error),
        }

        # Add per-class metrics
        for i in range(self.num_classes):
            metrics[f"precision_class_{i}"] = float(precision[i])
            metrics[f"recall_class_{i}"] = float(recall[i])
            metrics[f"f1_class_{i}"] = float(f1[i])

        return metrics

    def _compute_calibration_error(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error (ECE).
        
        Args:
            probs: Prediction probabilities (N, num_classes)
            labels: Ground truth labels (N,)
            n_bins: Number of bins for calibration
            
        Returns:
            Expected Calibration Error
        """
        # Get max probability and predicted class
        confidences = probs.max(axis=1)
        predictions = probs.argmax(axis=1)
        accuracies = (predictions == labels).astype(float)

        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _plot_confusion_matrix(self, cm: np.ndarray):
        """Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            
        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping visualization")
            return None

        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        return fig

    def evaluate_fairness(
        self,
        model: nn.Module,
        client_loaders: List,
        device: torch.device,
    ) -> Dict[str, any]:
        """Evaluate per-client fairness.
        
        Args:
            model: Neural network model
            client_loaders: List of DataLoaders for each client
            device: Device to run evaluation on
            
        Returns:
            Dictionary with fairness metrics
        """
        client_metrics = {}

        for client_id, loader in enumerate(client_loaders):
            metrics = self.evaluate_global_model(
                model, loader, device, save_visualizations=False
            )
            client_metrics[f"client_{client_id}"] = metrics

        # Compute fairness metrics
        accuracies = [m["accuracy"] for m in client_metrics.values()]
        f1_scores = [m["macro_f1"] for m in client_metrics.values()]

        fairness_metrics = {
            "client_metrics": client_metrics,
            "fairness_gap_accuracy": max(accuracies) - min(accuracies),
            "fairness_gap_f1": max(f1_scores) - min(f1_scores),
            "min_accuracy": min(accuracies),
            "max_accuracy": max(accuracies),
            "mean_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "min_f1": min(f1_scores),
            "max_f1": max(f1_scores),
            "mean_f1": np.mean(f1_scores),
            "std_f1": np.std(f1_scores),
        }

        return fairness_metrics

    def generate_classification_report(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
    ) -> str:
        """Generate detailed classification report.
        
        Args:
            labels: Ground truth labels
            preds: Predicted labels
            
        Returns:
            Classification report string
        """
        return classification_report(
            labels,
            preds,
            target_names=self.class_names,
            zero_division=0,
        )
