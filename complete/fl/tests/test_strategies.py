"""Unit tests for aggregation strategies."""

import pytest
import torch
from collections import OrderedDict

from fl.strategies.fedavg import FedAvgStrategy
from fl.strategies.fedprox import FedProxStrategy
from fl.strategies.fednova import FedNovaStrategy
from fl.strategies.scaffold import ScaffoldStrategy
from fl.strategies.fedadam import FedAdamStrategy


@pytest.fixture
def sample_params():
    """Create sample model parameters."""
    params1 = OrderedDict({
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
    })
    params2 = OrderedDict({
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
    })
    params3 = OrderedDict({
        "layer1.weight": torch.randn(10, 5),
        "layer1.bias": torch.randn(10),
    })
    return [params1, params2, params3]


@pytest.fixture
def sample_num_samples():
    """Create sample number of samples per client."""
    return [100, 150, 200]


def test_fedavg_aggregation(sample_params, sample_num_samples):
    """Test FedAvg aggregation."""
    strategy = FedAvgStrategy()
    aggregated = strategy.aggregate(sample_params, sample_num_samples)

    # Check that all parameters are present
    assert set(aggregated.keys()) == set(sample_params[0].keys())

    # Check that aggregated values are weighted averages
    total_samples = sum(sample_num_samples)
    expected_weight = sample_params[0]["layer1.weight"] * (100 / total_samples)
    expected_weight += sample_params[1]["layer1.weight"] * (150 / total_samples)
    expected_weight += sample_params[2]["layer1.weight"] * (200 / total_samples)

    assert torch.allclose(aggregated["layer1.weight"], expected_weight, atol=1e-5)


def test_fedprox_aggregation(sample_params, sample_num_samples):
    """Test FedProx aggregation (same as FedAvg on server side)."""
    strategy = FedProxStrategy(mu=0.01)
    aggregated = strategy.aggregate(sample_params, sample_num_samples)

    # FedProx server-side aggregation is same as FedAvg
    fedavg_strategy = FedAvgStrategy()
    fedavg_aggregated = fedavg_strategy.aggregate(sample_params, sample_num_samples)

    for key in aggregated.keys():
        assert torch.allclose(aggregated[key], fedavg_aggregated[key])


def test_fednova_aggregation(sample_params, sample_num_samples):
    """Test FedNova normalized aggregation."""
    strategy = FedNovaStrategy()
    local_steps = [10, 20, 15]

    aggregated = strategy.aggregate(
        sample_params,
        sample_num_samples,
        local_steps=local_steps,
    )

    # Check that all parameters are present
    assert set(aggregated.keys()) == set(sample_params[0].keys())


def test_scaffold_aggregation(sample_params, sample_num_samples):
    """Test SCAFFOLD aggregation."""
    strategy = ScaffoldStrategy()
    aggregated = strategy.aggregate(sample_params, sample_num_samples)

    # Check that all parameters are present
    assert set(aggregated.keys()) == set(sample_params[0].keys())

    # Check server controls are initialized
    assert strategy.server_controls is None or isinstance(strategy.server_controls, dict)


def test_fedadam_aggregation(sample_params, sample_num_samples):
    """Test FedAdam aggregation."""
    strategy = FedAdamStrategy(server_lr=0.01)

    # FedAdam needs global params
    global_params = OrderedDict({
        "layer1.weight": torch.zeros(10, 5),
        "layer1.bias": torch.zeros(10),
    })

    strategy.on_round_begin(0)
    aggregated = strategy.aggregate(
        sample_params,
        sample_num_samples,
        global_params=global_params,
    )

    # Check that all parameters are present
    assert set(aggregated.keys()) == set(sample_params[0].keys())

    # Check moments are initialized
    assert strategy.m is not None
    assert strategy.v is not None


def test_empty_client_params():
    """Test that strategies handle empty client list."""
    strategy = FedAvgStrategy()

    with pytest.raises(ValueError, match="No client parameters"):
        strategy.aggregate([], [])


def test_mismatched_lengths(sample_params):
    """Test that strategies handle mismatched input lengths."""
    strategy = FedAvgStrategy()

    with pytest.raises(ValueError, match="Mismatch"):
        strategy.aggregate(sample_params, [100, 150])  # Wrong length


def test_strategy_round_hooks(sample_params, sample_num_samples):
    """Test round begin/end hooks."""
    strategy = FedAvgStrategy()

    strategy.on_round_begin(1)
    assert strategy.round_num == 1

    aggregated = strategy.aggregate(sample_params, sample_num_samples)
    strategy.on_round_end(aggregated)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
