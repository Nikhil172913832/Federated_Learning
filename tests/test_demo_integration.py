"""Integration test for mnist_fedavg demo."""

import pytest
import subprocess
import sys
from pathlib import Path


@pytest.mark.slow
def test_mnist_fedavg_demo_runs():
    """Test that the killer demo runs successfully.
    
    This is an integration test that runs the actual demo script
    and verifies it completes without errors.
    """
    demo_path = Path(__file__).parent.parent / "examples" / "mnist_fedavg.py"
    
    # Run the demo with reduced rounds for faster testing
    result = subprocess.run(
        [sys.executable, str(demo_path)],
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )
    
    # Check it completed successfully
    assert result.returncode == 0, f"Demo failed with: {result.stderr}"
    
    # Check output contains expected strings
    assert "FEDERATED LEARNING DEMO" in result.stdout
    assert "Starting federated learning" in result.stdout
    assert "FINAL RESULTS" in result.stdout
    assert "Test Accuracy:" in result.stdout
    assert "completed successfully" in result.stdout
    
    # Check no errors in stderr
    assert "Error" not in result.stderr
    assert "Traceback" not in result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
