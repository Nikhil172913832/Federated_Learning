"""Baseline benchmarks for regression testing.

This module defines expected performance baselines for standard configurations.
Regression tests use these baselines to detect performance degradation.
"""

from typing import Dict, Any


# Baseline benchmarks for standard configurations
# These values should be updated when intentional improvements are made
BASELINES: Dict[str, Dict[str, Any]] = {
    "fedavg_mnist": {
        "accuracy": 0.985,
        "macro_f1": 0.983,
        "train_loss": 0.05,
        "rounds": 50,
        "time_sec": 180,
        "communication_mb": 45,
        "description": "FedAvg on MNIST with 10 clients, 50 rounds, IID data",
        "config": "config/examples/baseline_fedavg.yaml",
    },
    "fedavg_pneumoniamnist": {
        "accuracy": 0.92,
        "macro_f1": 0.91,
        "train_loss": 0.25,
        "rounds": 50,
        "time_sec": 200,
        "communication_mb": 40,
        "description": "FedAvg on PneumoniaMNIST with 10 clients, 50 rounds, IID data",
        "config": "config/default.yaml",
    },
    "fedprox_mnist_noniid": {
        "accuracy": 0.972,
        "macro_f1": 0.970,
        "train_loss": 0.08,
        "rounds": 100,
        "time_sec": 360,
        "communication_mb": 90,
        "description": "FedProx on MNIST with non-IID label skew (alpha=0.5)",
        "config": "config/examples/fedprox_heterogeneous.yaml",
    },
    "fedavg_mnist_dp": {
        "accuracy": 0.968,
        "macro_f1": 0.965,
        "train_loss": 0.12,
        "rounds": 50,
        "time_sec": 220,
        "communication_mb": 45,
        "privacy_epsilon": 3.0,
        "privacy_delta": 1e-5,
        "description": "FedAvg on MNIST with differential privacy (ε=3.0)",
        "config": "config/examples/dp_privacy.yaml",
    },
}


# Tolerance levels for regression detection
TOLERANCE = {
    "accuracy": 0.02,  # 2% tolerance
    "macro_f1": 0.02,
    "train_loss": 0.05,
    "time_sec": 0.10,  # 10% tolerance
    "communication_mb": 0.10,
}


def get_baseline(benchmark_name: str) -> Dict[str, Any]:
    """Get baseline metrics for a benchmark.
    
    Args:
        benchmark_name: Name of the benchmark (e.g., "fedavg_mnist")
        
    Returns:
        Dictionary containing baseline metrics
        
    Raises:
        KeyError: If benchmark_name not found
    """
    if benchmark_name not in BASELINES:
        available = ", ".join(BASELINES.keys())
        raise KeyError(
            f"Benchmark '{benchmark_name}' not found. "
            f"Available benchmarks: {available}"
        )
    return BASELINES[benchmark_name]


def check_regression(
    benchmark_name: str,
    actual_metrics: Dict[str, float],
    tolerance: Dict[str, float] = None,
) -> Dict[str, Any]:
    """Check if actual metrics represent a regression from baseline.
    
    Args:
        benchmark_name: Name of the benchmark
        actual_metrics: Dictionary of actual measured metrics
        tolerance: Optional custom tolerance levels
        
    Returns:
        Dictionary with regression check results:
        {
            "passed": bool,
            "failures": List[str],
            "comparisons": Dict[str, Dict]
        }
    """
    baseline = get_baseline(benchmark_name)
    tolerance = tolerance or TOLERANCE
    
    failures = []
    comparisons = {}
    
    for metric_name, baseline_value in baseline.items():
        if metric_name in ["description", "config"]:
            continue
            
        if metric_name not in actual_metrics:
            continue
            
        actual_value = actual_metrics[metric_name]
        tol = tolerance.get(metric_name, 0.0)
        
        # For accuracy and F1, higher is better
        if metric_name in ["accuracy", "macro_f1"]:
            threshold = baseline_value * (1 - tol)
            passed = actual_value >= threshold
            direction = "higher is better"
        # For loss and time, lower is better
        elif metric_name in ["train_loss", "time_sec", "communication_mb"]:
            threshold = baseline_value * (1 + tol)
            passed = actual_value <= threshold
            direction = "lower is better"
        # For privacy metrics, check equality within tolerance
        elif metric_name.startswith("privacy_"):
            threshold_low = baseline_value * (1 - tol)
            threshold_high = baseline_value * (1 + tol)
            passed = threshold_low <= actual_value <= threshold_high
            direction = "within tolerance"
        else:
            # Unknown metric, skip
            continue
        
        comparisons[metric_name] = {
            "baseline": baseline_value,
            "actual": actual_value,
            "threshold": threshold if not metric_name.startswith("privacy_") else (threshold_low, threshold_high),
            "passed": passed,
            "direction": direction,
        }
        
        if not passed:
            failures.append(
                f"{metric_name}: {actual_value:.4f} vs baseline {baseline_value:.4f} "
                f"(threshold: {threshold:.4f}, {direction})"
            )
    
    return {
        "passed": len(failures) == 0,
        "failures": failures,
        "comparisons": comparisons,
    }


def format_regression_report(result: Dict[str, Any]) -> str:
    """Format regression check result as a readable report.
    
    Args:
        result: Result from check_regression()
        
    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("REGRESSION TEST REPORT")
    lines.append("=" * 60)
    
    if result["passed"]:
        lines.append("✅ PASSED - No regressions detected")
    else:
        lines.append("❌ FAILED - Regressions detected")
        lines.append("")
        lines.append("Failures:")
        for failure in result["failures"]:
            lines.append(f"  - {failure}")
    
    lines.append("")
    lines.append("Detailed Comparisons:")
    lines.append("-" * 60)
    
    for metric_name, comparison in result["comparisons"].items():
        status = "✅" if comparison["passed"] else "❌"
        lines.append(
            f"{status} {metric_name:20s}: "
            f"actual={comparison['actual']:.4f}, "
            f"baseline={comparison['baseline']:.4f}, "
            f"{comparison['direction']}"
        )
    
    lines.append("=" * 60)
    return "\n".join(lines)
