"""Unit tests for client selection strategies."""

import pytest
import numpy as np

from fl.selection.random_selector import RandomSelector, UniformRandomSelector
from fl.selection.importance_sampling import ImportanceSamplingSelector
from fl.selection.cluster_based import ClusterBasedSelector
from fl.selection.fairness_aware import FairnessAwareSelector


@pytest.fixture
def num_clients():
    return 10


@pytest.fixture
def client_info():
    """Create sample client information."""
    return {
        i: {
            "data_size": np.random.randint(100, 500),
            "loss": np.random.rand(),
            "accuracy": np.random.rand(),
        }
        for i in range(10)
    }


def test_random_selector(num_clients):
    """Test random client selection."""
    selector = RandomSelector(num_clients, seed=42)
    selected = selector.select(num_select=5)

    assert len(selected) == 5
    assert len(set(selected)) == 5  # No duplicates
    assert all(0 <= cid < num_clients for cid in selected)


def test_random_selector_reproducibility(num_clients):
    """Test that random selection is reproducible with seed."""
    selector1 = RandomSelector(num_clients, seed=42)
    selector2 = RandomSelector(num_clients, seed=42)

    selected1 = selector1.select(num_select=5)
    selected2 = selector2.select(num_select=5)

    assert selected1 == selected2


def test_uniform_random_selector(num_clients):
    """Test uniform random selection cycles through clients."""
    selector = UniformRandomSelector(num_clients)

    # Select all clients over multiple rounds
    all_selected = []
    for _ in range(2):
        selected = selector.select(num_select=5)
        all_selected.extend(selected)

    # Check that all clients are selected exactly twice
    from collections import Counter
    counts = Counter(all_selected)
    assert len(counts) == num_clients
    assert all(count == 2 for count in counts.values())


def test_importance_sampling_with_data_size(num_clients, client_info):
    """Test importance sampling based on data size."""
    selector = ImportanceSamplingSelector(
        num_clients,
        sampling_method="data_size",
    )

    selected = selector.select(num_select=5, client_info=client_info)

    assert len(selected) == 5
    assert all(0 <= cid < num_clients for cid in selected)


def test_importance_sampling_without_info(num_clients):
    """Test importance sampling falls back to uniform without client info."""
    selector = ImportanceSamplingSelector(num_clients)

    selected = selector.select(num_select=5)

    assert len(selected) == 5


def test_fairness_aware_selector(num_clients):
    """Test fairness-aware selection."""
    selector = FairnessAwareSelector(
        num_clients,
        fairness_criterion="equal_participation",
    )

    # Run multiple rounds
    for _ in range(5):
        selected = selector.select(num_select=3)
        assert len(selected) == 3

    # Check fairness metrics
    metrics = selector.get_fairness_metrics()
    assert "variance" in metrics
    assert "gini" in metrics


def test_cluster_based_selector(num_clients, client_info):
    """Test cluster-based selection."""
    selector = ClusterBasedSelector(num_clients, num_clusters=3)

    selected = selector.select(num_select=6, client_info=client_info)

    assert len(selected) == 6
    assert all(0 <= cid < num_clients for cid in selected)


def test_selection_history_tracking(num_clients):
    """Test that selection history is tracked."""
    selector = RandomSelector(num_clients, seed=42)

    for _ in range(3):
        selector.select(num_select=5)

    assert len(selector.selection_history) == 3

    frequency = selector.get_selection_frequency()
    assert len(frequency) == num_clients
    assert sum(frequency.values()) == 15  # 3 rounds * 5 clients


def test_invalid_num_select(num_clients):
    """Test error when selecting more clients than available."""
    selector = RandomSelector(num_clients)

    with pytest.raises(ValueError):
        selector.select(num_select=num_clients + 1)


def test_fairness_metrics(num_clients):
    """Test fairness metrics calculation."""
    selector = RandomSelector(num_clients, seed=42)

    # Make some selections
    for _ in range(5):
        selector.select(num_select=3)

    metrics = selector.get_fairness_metrics()

    assert "variance" in metrics
    assert "gini" in metrics
    assert "min_count" in metrics
    assert "max_count" in metrics
    assert metrics["variance"] >= 0
    assert 0 <= metrics["gini"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
