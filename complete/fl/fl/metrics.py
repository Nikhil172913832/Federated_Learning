"""Prometheus metrics for FL platform monitoring."""

from prometheus_client import Counter, Gauge, Histogram, Summary
import time
from typing import Dict
import torch

# Training metrics
training_loss = Gauge(
    "fl_training_loss", "Training loss per client", ["client_id", "round"]
)

model_accuracy = Gauge(
    "fl_model_accuracy", "Model accuracy per client", ["client_id", "round"]
)

# Communication metrics
bytes_sent = Counter(
    "fl_bytes_sent_total", "Total bytes sent by clients", ["client_id"]
)

bytes_received = Counter(
    "fl_bytes_received_total", "Total bytes received by clients", ["client_id"]
)

compression_ratio = Gauge(
    "fl_compression_ratio", "Gradient compression ratio", ["client_id", "round"]
)

bandwidth_saved_bytes = Counter(
    "fl_bandwidth_saved_bytes", "Total bandwidth saved via compression", ["client_id"]
)

# Security metrics
byzantine_detected = Counter(
    "fl_byzantine_detected", "Number of Byzantine clients detected", ["round"]
)

anomaly_score = Gauge(
    "fl_anomaly_score", "Anomaly score for client updates", ["client_id", "round"]
)

# Privacy metrics
privacy_budget_consumed = Gauge(
    "fl_privacy_budget_consumed", "Privacy budget (epsilon) consumed", ["client_id"]
)

# System metrics
client_active = Gauge("fl_client_active", "Whether client is active", ["client_id"])

round_duration_seconds = Histogram(
    "fl_round_duration_seconds", "Duration of training round", ["round"]
)

aggregation_duration_seconds = Histogram(
    "fl_aggregation_duration_seconds", "Duration of aggregation", ["round", "method"]
)


class MetricsCollector:
    """Collect and export FL metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.start_time = time.time()

    def record_training_loss(self, client_id: str, round_num: int, loss: float):
        """Record training loss."""
        training_loss.labels(client_id=client_id, round=str(round_num)).set(loss)

    def record_accuracy(self, client_id: str, round_num: int, accuracy: float):
        """Record model accuracy."""
        model_accuracy.labels(client_id=client_id, round=str(round_num)).set(accuracy)

    def record_communication(
        self,
        client_id: str,
        sent_bytes: int,
        received_bytes: int,
        original_size: int,
        compressed_size: int,
        round_num: int,
    ):
        """Record communication metrics."""
        bytes_sent.labels(client_id=client_id).inc(sent_bytes)
        bytes_received.labels(client_id=client_id).inc(received_bytes)

        if compressed_size > 0:
            ratio = original_size / compressed_size
            compression_ratio.labels(client_id=client_id, round=str(round_num)).set(
                ratio
            )

            saved = original_size - compressed_size
            bandwidth_saved_bytes.labels(client_id=client_id).inc(saved)

    def record_byzantine_detection(self, round_num: int, count: int = 1):
        """Record Byzantine client detection."""
        byzantine_detected.labels(round=str(round_num)).inc(count)

    def record_anomaly_score(self, client_id: str, round_num: int, score: float):
        """Record anomaly score."""
        anomaly_score.labels(client_id=client_id, round=str(round_num)).set(score)

    def record_privacy_budget(self, client_id: str, epsilon: float):
        """Record privacy budget consumed."""
        privacy_budget_consumed.labels(client_id=client_id).set(epsilon)

    def set_client_active(self, client_id: str, active: bool):
        """Set client active status."""
        client_active.labels(client_id=client_id).set(1 if active else 0)

    def record_round_duration(self, round_num: int, duration: float):
        """Record round duration."""
        round_duration_seconds.labels(round=str(round_num)).observe(duration)

    def record_aggregation_duration(self, round_num: int, method: str, duration: float):
        """Record aggregation duration."""
        aggregation_duration_seconds.labels(
            round=str(round_num), method=method
        ).observe(duration)


# Global metrics collector instance
metrics_collector = MetricsCollector()
