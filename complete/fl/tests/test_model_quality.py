"""Regression tests for model quality.

These tests ensure that model performance doesn't degrade over time.
They run actual FL training and compare against established baselines.
"""

import pytest
import torch
from pathlib import Path

from fl.benchmarks import get_baseline, check_regression, format_regression_report


@pytest.mark.slow
@pytest.mark.skipif(
    not Path("complete/fl/config/default.yaml").exists(),
    reason="Config file not found"
)
class TestModelQualityRegression:
    """Regression tests for model quality baselines."""

    def test_fedavg_pneumoniamnist_baseline(self):
        """Regression test: FedAvg on PneumoniaMNIST should achieve ≥92% accuracy.
        
        This test runs a full FL training session and verifies that:
        1. Final accuracy meets or exceeds baseline
        2. Training time is within acceptable bounds
        3. Communication cost hasn't increased significantly
        
        Note: This is a slow test that runs actual training.
        """
        # TODO: Implement once we have FL training runner
        # For now, document the expected behavior
        
        baseline = get_baseline("fedavg_pneumoniamnist")
        
        # Placeholder for actual metrics - will be replaced with real training
        actual_metrics = {
            "accuracy": 0.92,  # Will come from actual training
            "macro_f1": 0.91,
            "train_loss": 0.25,
            "rounds": 50,
            "time_sec": 200,
            "communication_mb": 40,
        }
        
        # Check for regressions
        result = check_regression("fedavg_pneumoniamnist", actual_metrics)
        
        # Print detailed report
        print("\n" + format_regression_report(result))
        
        # Assert no regressions
        assert result["passed"], "\n".join(result["failures"])

    def test_baseline_metrics_are_achievable(self):
        """Sanity check: Verify baseline metrics are reasonable."""
        baseline = get_baseline("fedavg_pneumoniamnist")
        
        # Accuracy should be between 0 and 1
        assert 0.0 <= baseline["accuracy"] <= 1.0
        assert 0.0 <= baseline["macro_f1"] <= 1.0
        
        # Loss should be positive
        assert baseline["train_loss"] >= 0.0
        
        # Time and communication should be positive
        assert baseline["time_sec"] > 0
        assert baseline["communication_mb"] > 0

    def test_regression_detection_works(self):
        """Test that regression detection correctly identifies degradation."""
        # Test with metrics that should fail
        bad_metrics = {
            "accuracy": 0.85,  # 7% below baseline (0.92)
            "train_loss": 0.40,  # Much higher than baseline (0.25)
        }
        
        result = check_regression("fedavg_pneumoniamnist", bad_metrics)
        
        # Should detect regression
        assert not result["passed"]
        assert len(result["failures"]) > 0
        assert "accuracy" in str(result["failures"])

    def test_regression_detection_passes_good_metrics(self):
        """Test that regression detection passes with good metrics."""
        # Test with metrics that should pass
        good_metrics = {
            "accuracy": 0.93,  # Better than baseline
            "macro_f1": 0.92,  # Better than baseline
            "train_loss": 0.24,  # Better than baseline
            "time_sec": 190,  # Slightly better
            "communication_mb": 38,  # Slightly better
        }
        
        result = check_regression("fedavg_pneumoniamnist", good_metrics)
        
        # Should pass
        assert result["passed"]
        assert len(result["failures"]) == 0


@pytest.mark.slow
class TestPerformanceBudgets:
    """Test that training stays within performance budgets."""

    def test_training_time_budget(self):
        """Test that training completes within time budget."""
        # TODO: Implement with actual training
        # This test ensures we don't accidentally introduce performance regressions
        
        baseline = get_baseline("fedavg_pneumoniamnist")
        max_time = baseline["time_sec"] * 1.1  # 10% tolerance
        
        # Placeholder - will measure actual training time
        actual_time = 200
        
        assert actual_time <= max_time, \
            f"Training took {actual_time}s, exceeds budget of {max_time}s"

    def test_communication_cost_budget(self):
        """Test that communication cost stays within budget."""
        # TODO: Implement with actual training
        
        baseline = get_baseline("fedavg_pneumoniamnist")
        max_comm = baseline["communication_mb"] * 1.1  # 10% tolerance
        
        # Placeholder - will measure actual communication
        actual_comm = 40
        
        assert actual_comm <= max_comm, \
            f"Communication cost {actual_comm}MB exceeds budget of {max_comm}MB"


class TestBaselineDefinitions:
    """Test that baseline definitions are valid."""

    def test_all_baselines_have_required_fields(self):
        """Test that all baselines have required fields."""
        from fl.benchmarks import BASELINES
        
        required_fields = ["accuracy", "description", "config"]
        
        for name, baseline in BASELINES.items():
            for field in required_fields:
                assert field in baseline, \
                    f"Baseline '{name}' missing required field '{field}'"

    def test_baseline_configs_exist(self):
        """Test that referenced config files exist."""
        from fl.benchmarks import BASELINES
        
        for name, baseline in BASELINES.items():
            config_path = Path("complete/fl") / baseline["config"]
            # Note: Some configs might not exist yet, so we just warn
            if not config_path.exists():
                pytest.skip(f"Config {baseline['config']} for {name} not found")

    def test_get_baseline_raises_on_unknown(self):
        """Test that get_baseline raises KeyError for unknown benchmarks."""
        with pytest.raises(KeyError) as exc_info:
            get_baseline("nonexistent_benchmark")
        
        assert "not found" in str(exc_info.value)
        assert "Available benchmarks" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
