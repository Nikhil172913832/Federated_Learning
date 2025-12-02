"""TensorBoard integration for federated learning visualization."""

from typing import Dict, Optional, Any
from pathlib import Path

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class TensorBoardLogger:
    """TensorBoard logging wrapper for federated learning experiments."""

    def __init__(
        self,
        log_dir: str = "runs",
        experiment_name: str = "federated_learning",
        run_name: Optional[str] = None,
    ):
        """Initialize TensorBoard logger.

        Args:
            log_dir: Base directory for logs
            experiment_name: Experiment name
            run_name: Specific run name
        """
        if not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard not available")
            self.writer = None
            return

        # Create log directory path
        if run_name:
            full_path = Path(log_dir) / experiment_name / run_name
        else:
            full_path = Path(log_dir) / experiment_name

        self.writer = SummaryWriter(log_dir=str(full_path))
        self.log_dir = full_path

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int,
    ) -> None:
        """Log a scalar value.

        Args:
            tag: Name of the scalar
            value: Value to log
            step: Training step/round
        """
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: int,
    ) -> None:
        """Log multiple scalars under a main tag.

        Args:
            main_tag: Parent tag
            tag_scalar_dict: Dictionary of tag-value pairs
            step: Training step/round
        """
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(
        self,
        tag: str,
        values: Any,
        step: int,
    ) -> None:
        """Log a histogram.

        Args:
            tag: Name of the histogram
            values: Values (array-like)
            step: Training step/round
        """
        if self.writer:
            self.writer.add_histogram(tag, values, step)

    def log_image(
        self,
        tag: str,
        img_tensor: Any,
        step: int,
    ) -> None:
        """Log an image.

        Args:
            tag: Name of the image
            img_tensor: Image tensor
            step: Training step/round
        """
        if self.writer:
            self.writer.add_image(tag, img_tensor, step)

    def log_text(
        self,
        tag: str,
        text: str,
        step: int,
    ) -> None:
        """Log text.

        Args:
            tag: Text tag
            text: Text content
            step: Training step/round
        """
        if self.writer:
            self.writer.add_text(tag, text, step)

    def log_hparams(
        self,
        hparams: Dict[str, Any],
        metrics: Dict[str, float],
    ) -> None:
        """Log hyperparameters and metrics.

        Args:
            hparams: Hyperparameter dictionary
            metrics: Metrics dictionary
        """
        if self.writer:
            self.writer.add_hparams(hparams, metrics)

    def close(self) -> None:
        """Close the writer."""
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
