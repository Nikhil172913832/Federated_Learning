"""Unit tests for core FL client."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from core_fl.client import FederatedClient


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x):
        return self.fc(x)


def create_dummy_loader(num_samples=100, batch_size=10):
    """Create dummy data loader for testing."""
    images = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, 2, (num_samples,))
    
    # Create dataset that returns dict like HuggingFace
    class DictDataset:
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            return {"image": self.images[idx], "label": self.labels[idx]}
    
    dataset = DictDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size)


class TestFederatedClient:
    """Test FederatedClient class."""
    
    def test_init(self):
        """Test client initialization."""
        model = SimpleModel()
        loader = create_dummy_loader()
        
        client = FederatedClient(
            client_id=0,
            model=model,
            train_loader=loader,
        )
        
        assert client.client_id == 0
        assert client.model is model
        assert client.train_loader is loader
        assert client.dataset_size == 100
    
    def test_set_weights(self):
        """Test setting weights from global model."""
        model = SimpleModel()
        loader = create_dummy_loader()
        client = FederatedClient(0, model, loader)
        
        new_weights = {
            "fc.weight": torch.ones(2, 10) * 5.0,
            "fc.bias": torch.ones(2) * 5.0,
        }
        
        client.set_weights(new_weights)
        
        assert torch.allclose(
            client.model.fc.weight,
            torch.ones(2, 10) * 5.0
        )
    
    def test_train_returns_weights_and_loss(self):
        """Test that training returns weights and loss."""
        model = SimpleModel()
        loader = create_dummy_loader(num_samples=20, batch_size=10)
        client = FederatedClient(0, model, loader)
        
        weights, loss = client.train(epochs=1, lr=0.01)
        
        assert isinstance(weights, dict)
        assert "fc.weight" in weights
        assert "fc.bias" in weights
        assert isinstance(loss, float)
        assert loss > 0
    
    def test_train_with_fedprox(self):
        """Test training with FedProx proximal term."""
        model = SimpleModel()
        loader = create_dummy_loader(num_samples=20, batch_size=10)
        client = FederatedClient(0, model, loader)
        
        global_weights = {
            "fc.weight": torch.ones(2, 10),
            "fc.bias": torch.ones(2),
        }
        
        weights, loss = client.train(
            epochs=1,
            lr=0.01,
            global_weights=global_weights,
            mu=0.1,
        )
        
        assert isinstance(weights, dict)
        assert isinstance(loss, float)
    
    def test_evaluate(self):
        """Test client evaluation."""
        model = SimpleModel()
        loader = create_dummy_loader(num_samples=20, batch_size=10)
        client = FederatedClient(0, model, loader)
        
        loss, accuracy = client.evaluate()
        
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
