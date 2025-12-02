"""Enhanced experiment tracking for federated learning.

Supports MLflow and Weights & Biases for comprehensive experiment tracking.
"""

import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, Optional, Any

try:
    import mlflow  # type: ignore
    MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    mlflow = None  # type: ignore
    MLFLOW_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ExperimentTracker:
    """Unified experiment tracking interface supporting multiple backends."""

    def __init__(
        self,
        experiment_name: str = "federated_learning",
        run_name: Optional[str] = None,
        use_mlflow: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
    ):
        """Initialize experiment tracker.

        Args:
            experiment_name: Name of the experiment
            run_name: Name of this specific run
            use_mlflow: Whether to use MLflow
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            wandb_entity: W&B entity/team name
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.mlflow_run = None

        # Initialize MLflow
        if self.use_mlflow:
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
            mlflow.set_experiment(experiment_name)

        # Initialize W&B
        if self.use_wandb:
            if not WANDB_AVAILABLE:
                print("Warning: wandb not available, disabling W&B tracking")
                self.use_wandb = False
            else:
                wandb.init(
                    project=wandb_project or experiment_name,
                    entity=wandb_entity,
                    name=run_name,
                    reinit=True,
                )

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters.

        Args:
            params: Dictionary of parameters
        """
        if self.use_mlflow:
            try:
                mlflow.log_params(params)
            except Exception as e:
                print(f"MLflow log_params error: {e}")

        if self.use_wandb:
            wandb.config.update(params)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics.

        Args:
            metrics: Dictionary of metric values
            step: Step/epoch number
        """
        if self.use_mlflow:
            try:
                mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                print(f"MLflow log_metrics error: {e}")

        if self.use_wandb:
            wandb.log(metrics, step=step)

    def log_artifact(self, artifact_path: str) -> None:
        """Log an artifact file.

        Args:
            artifact_path: Path to artifact file
        """
        if self.use_mlflow:
            mlflow.log_artifact(artifact_path)

        if self.use_wandb:
            wandb.save(artifact_path)

    def finish(self) -> None:
        """Finish the experiment run."""
        if self.use_wandb:
            wandb.finish()

    def __enter__(self):
        """Context manager entry."""
        if self.use_mlflow:
            self.mlflow_run = mlflow.start_run(run_name=self.run_name)
            self.mlflow_run.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.use_mlflow and self.mlflow_run:
            self.mlflow_run.__exit__(exc_type, exc_val, exc_tb)
        self.finish()


# Legacy functions for backward compatibility
@contextmanager
def start_run(experiment: str, run_name: Optional[str] = None) -> Iterator[None]:
    if mlflow is None:
        yield
        return
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns"))
    
    # Use consistent experiment name for grouping
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name):
        yield


def log_params(params: Dict):
    if mlflow is None:
        return
    mlflow.log_params(params)


def log_metrics(metrics: Dict, step: Optional[int] = None):
    if mlflow is None:
        return
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: Path, artifact_path: Optional[str] = None):
    if mlflow is None:
        return
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


