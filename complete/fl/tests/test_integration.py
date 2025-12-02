"""Integration tests for federated learning workflow."""

import pytest
import torch
import torch.nn as nn
from collections import OrderedDict

from fl.core.client import FederatedClient
from fl.core.server import FederatedServer
from fl.models.cnn import SimpleCNN
from fl.strategies.fedavg import FedAvgStrategy


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def model():
    return SimpleModel()


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader."""
    class MockBatch:
        def __init__(self):
            self.data = {
                "image": torch.randn(4, 10),
                "label": torch.randint(0, 2, (4,)),
            }

        def __getitem__(self, key):
            return self.data[key]

    class MockDataLoader:
        def __init__(self):
            self.dataset = list(range(20))

        def __iter__(self):
            for _ in range(5):
                yield MockBatch()

    return MockDataLoader()


def test_client_training(model, device, mock_dataloader):
    """Test client training process."""
    client = FederatedClient(
        client_id=0,
        model=model,
        device=device,
    )

    # Train for one epoch
    updated_params, loss, num_samples = client.train(
        trainloader=mock_dataloader,
        epochs=1,
        lr=0.01,
    )

    assert isinstance(updated_params, dict)
    assert isinstance(loss, float)
    assert num_samples == 20


def test_client_evaluation(model, device, mock_dataloader):
    """Test client evaluation."""
    client = FederatedClient(
        client_id=0,
        model=model,
        device=device,
    )

    loss, accuracy, num_samples = client.evaluate(mock_dataloader)

    assert isinstance(loss, float)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1
    assert num_samples == 20


def test_server_aggregation(model):
    """Test server aggregation."""
    server = FederatedServer(
        model=model,
        strategy="fedavg",
    )

    # Create mock client parameters
    client_params = [
        OrderedDict({
            "fc.weight": torch.randn(2, 10),
            "fc.bias": torch.randn(2),
        })
        for _ in range(3)
    ]
    num_samples = [100, 150, 200]

    # Aggregate
    aggregated = server.aggregate_parameters(client_params, num_samples)

    assert isinstance(aggregated, dict)
    assert "fc.weight" in aggregated
    assert "fc.bias" in aggregated


def test_federated_round(model, device, mock_dataloader):
    """Test a complete federated learning round."""
    # Initialize server
    server = FederatedServer(model=SimpleModel(), strategy="fedavg")
    global_params = server.get_global_parameters()

    # Initialize clients
    num_clients = 3
    clients = [
        FederatedClient(
            client_id=i,
            model=SimpleModel(),
            device=device,
        )
        for i in range(num_clients)
    ]

    # Distribute global model
    for client in clients:
        client.set_parameters(global_params)

    # Client training
    client_params = []
    num_samples = []

    for client in clients:
        params, loss, n_samples = client.train(
            trainloader=mock_dataloader,
            epochs=1,
            lr=0.01,
        )
        client_params.append(params)
        num_samples.append(n_samples)

    # Server aggregation
    aggregated = server.aggregate_parameters(client_params, num_samples)
    server.update_global_model(aggregated)

    # Check round number updated
    assert server.round_num == 1


def test_multiple_rounds(model, device, mock_dataloader):
    """Test multiple federated learning rounds."""
    server = FederatedServer(model=SimpleModel(), strategy="fedavg")

    num_rounds = 3
    num_clients = 2

    for round_num in range(num_rounds):
        global_params = server.get_global_parameters()

        # Client training
        client_params = []
        num_samples = []

        for client_id in range(num_clients):
            client = FederatedClient(
                client_id=client_id,
                model=SimpleModel(),
                device=device,
            )
            client.set_parameters(global_params)

            params, loss, n_samples = client.train(
                trainloader=mock_dataloader,
                epochs=1,
                lr=0.01,
            )
            client_params.append(params)
            num_samples.append(n_samples)

        # Aggregate and update
        aggregated = server.aggregate_parameters(client_params, num_samples)
        server.update_global_model(aggregated)

        # Log metrics
        server.log_round_metrics({
            "round": round_num,
            "avg_loss": sum([1.0] * num_clients) / num_clients,
        })

    assert server.round_num == num_rounds
    assert len(server.metrics_history) == num_rounds


def test_model_convergence(device, mock_dataloader):
    """Test that training actually changes model parameters."""
    model1 = SimpleModel()
    model2 = SimpleModel()

    # Copy initial weights
    model2.load_state_dict(model1.state_dict())

    client = FederatedClient(
        client_id=0,
        model=model1,
        device=device,
    )

    # Train
    client.train(trainloader=mock_dataloader, epochs=1, lr=0.01)

    # Weights should have changed
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert not torch.allclose(p1, p2), f"Parameter {n1} did not change"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
