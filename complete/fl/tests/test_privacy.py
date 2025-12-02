"""Unit tests for privacy modules."""

import pytest
import torch
import torch.nn as nn
from collections import OrderedDict

from fl.privacy.secure_aggregation import SecureAggregator
from fl.privacy.byzantine_robust import ByzantineRobustAggregator, AnomalyDetector


@pytest.fixture
def sample_params():
    """Create sample model parameters."""
    params1 = OrderedDict({
        "layer.weight": torch.randn(5, 3),
        "layer.bias": torch.randn(5),
    })
    params2 = OrderedDict({
        "layer.weight": torch.randn(5, 3),
        "layer.bias": torch.randn(5),
    })
    params3 = OrderedDict({
        "layer.weight": torch.randn(5, 3),
        "layer.bias": torch.randn(5),
    })
    return [params1, params2, params3]


def test_secure_aggregator_key_generation():
    """Test secure aggregator key generation."""
    aggregator = SecureAggregator(num_clients=3)

    key1 = aggregator.generate_client_key(client_id=0, seed=42)
    key2 = aggregator.generate_client_key(client_id=1, seed=43)

    assert key1 is not None
    assert key2 is not None
    assert key1 != key2


def test_secure_aggregator_masking():
    """Test parameter masking and aggregation."""
    aggregator = SecureAggregator(num_clients=3)

    # Generate keys
    for i in range(3):
        aggregator.generate_client_key(client_id=i, seed=42 + i)

    # Create sample parameters
    params = OrderedDict({
        "weight": torch.ones(3, 3),
    })

    # Mask parameters
    masked = aggregator.mask_parameters(
        params,
        client_id=0,
        other_client_ids=[1, 2],
    )

    assert "weight" in masked
    # Masked params should be different from original
    assert not torch.allclose(masked["weight"], params["weight"])


def test_byzantine_krum():
    """Test Krum Byzantine-robust aggregation."""
    aggregator = ByzantineRobustAggregator(strategy="krum")

    # Create honest and malicious clients
    honest_param = OrderedDict({
        "weight": torch.ones(5, 3),
    })

    client_params = [
        honest_param,
        honest_param.copy(),
        OrderedDict({"weight": torch.ones(5, 3) * 100}),  # Malicious
    ]

    num_samples = [100, 100, 100]

    # Krum should select honest update
    result = aggregator.aggregate(
        client_params,
        num_samples,
        num_byzantine=1,
    )

    # Result should be close to honest params
    assert torch.allclose(result["weight"], honest_param["weight"], atol=10.0)


def test_byzantine_trimmed_mean(sample_params):
    """Test trimmed mean aggregation."""
    aggregator = ByzantineRobustAggregator(strategy="trimmed_mean")

    result = aggregator.aggregate(
        sample_params,
        [100, 100, 100],
        num_byzantine=1,
    )

    assert "layer.weight" in result
    assert "layer.bias" in result


def test_byzantine_median(sample_params):
    """Test coordinate-wise median aggregation."""
    aggregator = ByzantineRobustAggregator(strategy="median")

    result = aggregator.aggregate(
        sample_params,
        [100, 100, 100],
    )

    assert "layer.weight" in result
    assert "layer.bias" in result


def test_byzantine_bulyan():
    """Test Bulyan aggregation."""
    aggregator = ByzantineRobustAggregator(strategy="bulyan")

    # Need enough clients for Bulyan
    params = [
        OrderedDict({"weight": torch.randn(3, 3)})
        for _ in range(7)
    ]
    num_samples = [100] * 7

    result = aggregator.aggregate(
        params,
        num_samples,
        num_byzantine=2,
    )

    assert "weight" in result


def test_anomaly_detector(sample_params):
    """Test anomaly detection."""
    detector = AnomalyDetector(threshold=2.0)

    # Add an anomalous update
    anomalous = OrderedDict({
        "layer.weight": torch.randn(5, 3) * 100,  # Very large
        "layer.bias": torch.randn(5) * 100,
    })
    all_params = sample_params + [anomalous]

    anomalies = detector.detect_anomalies(all_params)

    assert len(anomalies) == 4
    # Last one should be detected as anomaly
    assert anomalies[-1] == True


def test_anomaly_detector_all_normal(sample_params):
    """Test anomaly detector with all normal updates."""
    detector = AnomalyDetector(threshold=3.0)

    anomalies = detector.detect_anomalies(sample_params)

    # All should be normal
    assert all(not a for a in anomalies)


def test_invalid_strategy():
    """Test invalid Byzantine strategy."""
    aggregator = ByzantineRobustAggregator(strategy="invalid")

    with pytest.raises(ValueError, match="Unknown strategy"):
        aggregator.aggregate([], [], num_byzantine=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
