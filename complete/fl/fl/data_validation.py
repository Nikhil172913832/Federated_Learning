"""Data validation and quality monitoring.

This module provides tools for:
- Schema validation
- Data quality metrics
- Distribution drift detection
- Automated alerts for quality issues
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any
from pathlib import Path

import numpy as np
import torch

try:
    import yaml
except ImportError:
    yaml = None


logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Data quality metrics report.
    
    Attributes:
        num_samples: Total number of samples
        class_distribution: Distribution of samples per class
        missing_values: Number of missing values
        outliers: Number of outliers detected
        drift_score: KL divergence from baseline distribution
        mean_pixel_value: Mean pixel value across dataset
        std_pixel_value: Standard deviation of pixel values
    """
    num_samples: int
    class_distribution: Dict[int, int]
    missing_values: int
    outliers: int
    drift_score: float
    mean_pixel_value: float
    std_pixel_value: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "num_samples": self.num_samples,
            "class_distribution": self.class_distribution,
            "missing_values": self.missing_values,
            "outliers": self.outliers,
            "drift_score": self.drift_score,
            "mean_pixel_value": self.mean_pixel_value,
            "std_pixel_value": self.std_pixel_value,
        }


class DataValidator:
    """Validator for data quality and schema compliance."""

    def __init__(self, schema_path: Optional[Path] = None):
        """Initialize validator.
        
        Args:
            schema_path: Path to schema YAML file
        """
        self.schema = self._load_schema(schema_path) if schema_path else self._default_schema()
        self.baseline_stats = None

    def _default_schema(self) -> Dict[str, Any]:
        """Default schema for medical imaging data."""
        return {
            "image_shape": [28, 28],
            "num_classes": 10,
            "expected_dtype": "float32",
            "label_dtype": "int64",
            "min_samples_per_partition": 100,
            "max_class_imbalance": 0.3,
            "pixel_value_range": [-5.0, 5.0],  # After normalization
        }

    def _load_schema(self, schema_path: Path) -> Dict[str, Any]:
        """Load schema from YAML file.
        
        Args:
            schema_path: Path to schema file
            
        Returns:
            Schema dictionary
        """
        if not schema_path.exists():
            logger.warning(f"Schema file not found: {schema_path}, using defaults")
            return self._default_schema()

        if yaml is None:
            logger.warning("PyYAML not available, using default schema")
            return self._default_schema()

        with open(schema_path, 'r') as f:
            schema = yaml.safe_load(f)

        return schema

    def validate_partition(
        self,
        partition_id: int,
        dataloader,
        set_baseline: bool = False,
    ) -> DataQualityReport:
        """Validate data quality for a partition.
        
        Args:
            partition_id: ID of the partition
            dataloader: DataLoader for the partition
            set_baseline: Whether to set this as baseline for drift detection
            
        Returns:
            DataQualityReport with quality metrics
        """
        logger.info(f"Validating partition {partition_id}")

        # Collect data
        all_images = []
        all_labels = []

        for batch in dataloader:
            all_images.append(batch["image"])
            all_labels.append(batch["label"])

        images = torch.cat(all_images, dim=0)
        labels = torch.cat(all_labels, dim=0)

        # Schema validation
        self._validate_schema(images, labels)

        # Compute quality metrics
        class_dist = self._compute_class_distribution(labels)
        outliers = self._detect_outliers(images)
        drift = self._compute_drift(class_dist, set_baseline)

        # Compute statistics
        mean_pixel = float(images.mean())
        std_pixel = float(images.std())

        report = DataQualityReport(
            num_samples=len(images),
            class_distribution=class_dist,
            missing_values=0,  # PyTorch tensors don't have missing values
            outliers=outliers,
            drift_score=drift,
            mean_pixel_value=mean_pixel,
            std_pixel_value=std_pixel,
        )

        # Alert on quality issues
        self._check_quality_thresholds(report)

        return report

    def _validate_schema(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        """Validate data against schema.
        
        Args:
            images: Image tensor
            labels: Label tensor
            
        Raises:
            ValueError: If schema validation fails
        """
        # Check image shape
        expected_shape = self.schema["image_shape"]
        actual_shape = list(images.shape[2:])  # Skip batch and channel dims
        
        if actual_shape != expected_shape:
            raise ValueError(
                f"Image shape mismatch: expected {expected_shape}, got {actual_shape}"
            )

        # Check label range
        num_classes = self.schema["num_classes"]
        if labels.min() < 0 or labels.max() >= num_classes:
            raise ValueError(
                f"Labels out of range [0, {num_classes}): "
                f"min={labels.min()}, max={labels.max()}"
            )

        # Check pixel value range
        pixel_range = self.schema["pixel_value_range"]
        if images.min() < pixel_range[0] or images.max() > pixel_range[1]:
            logger.warning(
                f"Pixel values outside expected range {pixel_range}: "
                f"min={images.min():.3f}, max={images.max():.3f}"
            )

        logger.info("✅ Schema validation passed")

    def _compute_class_distribution(self, labels: torch.Tensor) -> Dict[int, int]:
        """Compute class distribution.
        
        Args:
            labels: Label tensor
            
        Returns:
            Dictionary mapping class to count
        """
        unique, counts = torch.unique(labels, return_counts=True)
        return {int(c): int(count) for c, count in zip(unique, counts)}

    def _detect_outliers(self, images: torch.Tensor, threshold: float = 3.0) -> int:
        """Detect outliers using IQR method.
        
        Args:
            images: Image tensor
            threshold: Number of IQRs for outlier detection
            
        Returns:
            Number of outliers detected
        """
        # Compute per-image mean
        image_means = images.mean(dim=(1, 2, 3))

        # Compute IQR
        q1 = torch.quantile(image_means, 0.25)
        q3 = torch.quantile(image_means, 0.75)
        iqr = q3 - q1

        # Detect outliers
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        outliers = ((image_means < lower_bound) | (image_means > upper_bound)).sum()

        return int(outliers)

    def _compute_drift(
        self,
        class_dist: Dict[int, int],
        set_baseline: bool = False,
    ) -> float:
        """Compute distribution drift using KL divergence.
        
        Args:
            class_dist: Current class distribution
            set_baseline: Whether to set this as baseline
            
        Returns:
            KL divergence from baseline (0 if no baseline)
        """
        if set_baseline or self.baseline_stats is None:
            self.baseline_stats = class_dist
            logger.info("Set baseline distribution for drift detection")
            return 0.0

        # Convert to probability distributions
        total = sum(class_dist.values())
        current_probs = np.array([class_dist.get(i, 0) / total for i in range(self.schema["num_classes"])])

        baseline_total = sum(self.baseline_stats.values())
        baseline_probs = np.array([self.baseline_stats.get(i, 0) / baseline_total for i in range(self.schema["num_classes"])])

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        current_probs = current_probs + epsilon
        baseline_probs = baseline_probs + epsilon

        # Compute KL divergence
        kl_div = np.sum(current_probs * np.log(current_probs / baseline_probs))

        return float(kl_div)

    def _check_quality_thresholds(self, report: DataQualityReport) -> None:
        """Check quality metrics against thresholds and alert.
        
        Args:
            report: Data quality report
        """
        # Check minimum samples
        min_samples = self.schema.get("min_samples_per_partition", 100)
        if report.num_samples < min_samples:
            logger.warning(
                f"⚠️  Low sample count: {report.num_samples} < {min_samples}"
            )

        # Check class imbalance
        max_imbalance = self.schema.get("max_class_imbalance", 0.3)
        if report.class_distribution:
            counts = list(report.class_distribution.values())
            imbalance = (max(counts) - min(counts)) / max(counts)
            if imbalance > max_imbalance:
                logger.warning(
                    f"⚠️  High class imbalance: {imbalance:.3f} > {max_imbalance}"
                )

        # Check drift
        if report.drift_score > 0.3:
            logger.warning(
                f"⚠️  High distribution drift detected: {report.drift_score:.3f}"
            )

        # Check outliers
        outlier_ratio = report.outliers / report.num_samples
        if outlier_ratio > 0.05:  # More than 5% outliers
            logger.warning(
                f"⚠️  High outlier ratio: {outlier_ratio:.2%}"
            )
