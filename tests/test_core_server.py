"""Unit tests for core FL server."""

import pytest
import torch
import torch.nn as nn
from core_fl.server import FederatedServer


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


class TestFederatedServer:
    """Test FederatedServer class."""
    
    def test_init(self):
        """Test server initialization."""
        model = SimpleModel()
        server = FederatedServer(model, strategy="fedavg")
        
        assert server.global_model is model
        assert server.strategy == "fedavg"
        assert server.round_num == 0
    
    def test_get_global_weights(self):
        """Test getting global weights."""
        model = SimpleModel()
        server = FederatedServer(model)
        
        weights = server.get_global_weights()
        
        assert isinstance(weights, dict)
        assert "fc.weight" in weights
        assert "fc.bias" in weights
    
    def test_fedavg_aggregate(self):
        """Test FedAvg aggregation."""
        model = SimpleModel()
        server = FederatedServer(model, strategy="fedavg")
        
        # Create mock client weights
        client1_weights = {
            "fc.weight": torch.ones(2, 10) * 1.0,
            "fc.bias": torch.ones(2) * 1.0,
        }
        client2_weights = {
            "fc.weight": torch.ones(2, 10) * 2.0,
            "fc.bias": torch.ones(2) * 2.0,
        }
        
        client_weights = [client1_weights, client2_weights]
        client_sizes = [100, 200]  # Client 2 has 2x data
        
        # Aggregate
        aggregated = server.aggregate_weights(client_weights, client_sizes)
        
        # Expected: (1*100 + 2*200) / 300 = 500/300 = 1.667
        expected_value = (1.0 * 100 + 2.0 * 200) / 300
        
        assert torch.allclose(
            aggregated["fc.weight"],
            torch.ones(2, 10) * expected_value,
            atol=1e-5
        )
        assert torch.allclose(
            aggregated["fc.bias"],
            torch.ones(2) * expected_value,
            atol=1e-5
        )
    
    def test_update_global_model(self):
        """Test updating global model."""
        model = SimpleModel()
        server = FederatedServer(model)
        
        new_weights = {
            "fc.weight": torch.ones(2, 10) * 5.0,
            "fc.bias": torch.ones(2) * 5.0,
        }
        
        server.update_global_model(new_weights)
        
        assert torch.allclose(
            server.global_model.fc.weight,
            torch.ones(2, 10) * 5.0
        )
        assert server.round_num == 1
    
    def test_unknown_strategy_raises(self):
        """Test that unknown strategy raises error."""
        model = SimpleModel()
        server = FederatedServer(model, strategy="unknown")
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            server.aggregate_weights([], [])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
