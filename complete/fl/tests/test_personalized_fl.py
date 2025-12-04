"""
Tests for personalized FL with knowledge distillation.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import tempfile

from fl.personalization import (
    DistillationConfig,
    DistillationLoss,
    LocalDistillationTrainer,
    LoRALayer,
    LoRALinear,
    BottleneckAdapter,
    ModelWithAdapters,
    PersonalizedFLClient,
    PersonalizedClientConfig,
    PersonalizedFLServer,
    PersonalizedServerConfig,
    create_personalized_fl_config,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def dummy_data():
    """Create dummy dataset."""
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=10)


class TestDistillation:
    """Test knowledge distillation components."""
    
    def test_distillation_config(self):
        """Test distillation config creation."""
        config = DistillationConfig(
            temperature=2.0,
            lambda_kd=0.5,
            kd_loss_type="kl"
        )
        assert config.temperature == 2.0
        assert config.lambda_kd == 0.5
        assert config.kd_loss_type == "kl"
        
    def test_distillation_loss_kl(self, device):
        """Test KL divergence distillation loss."""
        config = DistillationConfig(
            temperature=2.0,
            lambda_kd=0.5,
            lambda_ce=0.5,
            kd_loss_type="kl"
        )
        loss_fn = DistillationLoss(config)
        
        student_logits = torch.randn(8, 10)
        teacher_logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))
        
        loss, losses = loss_fn(student_logits, teacher_logits, labels)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert "ce" in losses
        assert "kd" in losses
        assert "total" in losses
        
    def test_distillation_loss_mse(self, device):
        """Test MSE distillation loss."""
        config = DistillationConfig(
            temperature=2.0,
            lambda_kd=0.5,
            lambda_ce=0.5,
            kd_loss_type="mse"
        )
        loss_fn = DistillationLoss(config)
        
        student_logits = torch.randn(8, 10)
        teacher_logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))
        
        loss, losses = loss_fn(student_logits, teacher_logits, labels)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        
    def test_local_distillation_trainer(self, simple_model, dummy_data, device):
        """Test local distillation trainer."""
        teacher = SimpleModel()
        student = SimpleModel()
        
        config = DistillationConfig(temperature=2.0, lambda_kd=0.5)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
        
        trainer = LocalDistillationTrainer(
            student, teacher, config, optimizer, device
        )
        
        # Train one epoch
        metrics = trainer.train_epoch(dummy_data)
        
        assert "total" in metrics
        assert "ce" in metrics
        assert "kd" in metrics
        assert metrics["total"] > 0


class TestAdapters:
    """Test adapter modules."""
    
    def test_lora_layer(self):
        """Test LoRA layer."""
        lora = LoRALayer(
            in_features=10,
            out_features=20,
            rank=4,
            alpha=16.0
        )
        
        x = torch.randn(8, 10)
        output = lora(x)
        
        assert output.shape == (8, 20)
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad
        
    def test_lora_linear(self):
        """Test LoRA linear layer."""
        base = nn.Linear(10, 20)
        lora_linear = LoRALinear(
            base, rank=4, alpha=16.0, freeze_base=True
        )
        
        # Check base is frozen
        assert not base.weight.requires_grad
        
        x = torch.randn(8, 10)
        output = lora_linear(x)
        
        assert output.shape == (8, 20)
        
    def test_bottleneck_adapter(self):
        """Test bottleneck adapter."""
        adapter = BottleneckAdapter(
            input_dim=20,
            bottleneck_dim=8,
            activation="relu"
        )
        
        x = torch.randn(8, 20)
        output = adapter(x)
        
        # Should have residual connection
        assert output.shape == x.shape
        
    def test_model_with_adapters(self, simple_model):
        """Test adding adapters to model."""
        adapter_config = {
            "type": "lora",
            "rank": 4,
            "alpha": 16.0
        }
        
        adapted = ModelWithAdapters(
            simple_model,
            adapter_config,
            target_modules=None  # Auto-detect
        )
        
        # Check adapters were added
        adapter_params = adapted.get_adapter_params()
        assert len(adapter_params) > 0
        
        # Check forward pass
        x = torch.randn(4, 10)
        output = adapted(x)
        assert output.shape == (4, 2)


class TestPersonalizedClient:
    """Test personalized FL client."""
    
    def test_client_initialization(self, simple_model, device):
        """Test client initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PersonalizedClientConfig(
                client_id="test_client",
                adapter_type="lora",
                adapter_rank=4,
                storage_path=Path(tmpdir)
            )
            
            client = PersonalizedFLClient(config, simple_model, device)
            
            assert client.config.client_id == "test_client"
            assert client.device == device
            
    def test_client_receive_global_model(self, simple_model, device):
        """Test receiving global model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PersonalizedClientConfig(
                client_id="test_client",
                adapter_type="lora",
                storage_path=Path(tmpdir)
            )
            
            client = PersonalizedFLClient(config, simple_model, device)
            
            # Create fake global weights
            global_weights = {
                name: param.detach().clone()
                for name, param in simple_model.named_parameters()
            }
            
            client.receive_global_model(global_weights, round_idx=0)
            
            assert client.round_idx == 0
            
    def test_client_local_train(self, simple_model, dummy_data, device):
        """Test local training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PersonalizedClientConfig(
                client_id="test_client",
                adapter_type="lora",
                local_epochs=1,
                learning_rate=0.01,
                storage_path=Path(tmpdir),
                distillation_config=DistillationConfig()
            )
            
            client = PersonalizedFLClient(config, simple_model, device)
            client.set_dataset_size(100)
            
            # Send global model
            global_weights = {
                name: param.detach().clone()
                for name, param in simple_model.named_parameters()
            }
            client.receive_global_model(global_weights, round_idx=0)
            
            # Train
            metrics = client.local_train(dummy_data)
            
            assert "total" in metrics or "val_accuracy" in metrics
            
    def test_client_get_update(self, simple_model, device):
        """Test getting model update."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PersonalizedClientConfig(
                client_id="test_client",
                adapter_type="lora",
                storage_path=Path(tmpdir)
            )
            
            client = PersonalizedFLClient(config, simple_model, device)
            
            update = client.get_model_update()
            
            # Should not include adapter params
            for name in update.keys():
                assert "lora" not in name
                assert "adapter" not in name


class TestPersonalizedServer:
    """Test personalized FL server."""
    
    def test_server_initialization(self, simple_model, device):
        """Test server initialization."""
        config = PersonalizedServerConfig(
            total_rounds=10,
            clients_per_round=5
        )
        
        server = PersonalizedFLServer(config, simple_model, device)
        
        assert server.config.total_rounds == 10
        assert server.current_round == 0
        
    def test_server_register_client(self, simple_model, device):
        """Test client registration."""
        config = PersonalizedServerConfig()
        server = PersonalizedFLServer(config, simple_model, device)
        
        server.register_client("client_1", {"dataset_size": 100})
        
        assert "client_1" in server.registered_clients
        assert server.registered_clients["client_1"]["total_samples"] == 100
        
    def test_server_select_clients(self, simple_model, device):
        """Test client selection."""
        config = PersonalizedServerConfig(clients_per_round=3)
        server = PersonalizedFLServer(config, simple_model, device)
        
        # Register clients
        for i in range(10):
            server.register_client(f"client_{i}", {"dataset_size": 100})
            
        # Select clients
        selected = server.select_clients(round_idx=0)
        
        assert len(selected) == 3
        assert all(cid in server.registered_clients for cid in selected)
        
    def test_server_aggregate_fedavg(self, simple_model, device):
        """Test FedAvg aggregation."""
        config = PersonalizedServerConfig(
            aggregation_strategy="fedavg",
            weighted_aggregation=False
        )
        server = PersonalizedFLServer(config, simple_model, device)
        
        # Create fake client updates
        client_updates = {}
        for i in range(3):
            update = {
                name: torch.randn_like(param)
                for name, param in simple_model.named_parameters()
            }
            client_updates[f"client_{i}"] = update
            
        # Aggregate
        stats = server.aggregate_updates(
            client_updates,
            client_weights=None,
            round_idx=0
        )
        
        assert stats["num_clients"] == 3
        assert "mean_update_norm" in stats
        
    def test_server_anomaly_detection(self, simple_model, device):
        """Test anomaly detection."""
        config = PersonalizedServerConfig(
            enable_anomaly_detection=True,
            anomaly_threshold_multiplier=2.0
        )
        server = PersonalizedFLServer(config, simple_model, device)
        
        # Create updates: 2 normal, 1 anomalous
        client_updates = {}
        for i in range(2):
            update = {
                name: torch.randn_like(param) * 0.1
                for name, param in simple_model.named_parameters()
            }
            client_updates[f"client_{i}"] = update
            
        # Anomalous update (10x larger)
        anomalous_update = {
            name: torch.randn_like(param) * 10.0
            for name, param in simple_model.named_parameters()
        }
        client_updates["anomalous_client"] = anomalous_update
        
        # Aggregate (should filter anomaly)
        stats = server.aggregate_updates(
            client_updates,
            client_weights=None,
            round_idx=0
        )
        
        # Should have filtered the anomalous client
        assert stats["num_clients"] <= 2


class TestOrchestrator:
    """Test orchestrator."""
    
    def test_create_config(self):
        """Test config creation helper."""
        config = create_personalized_fl_config(
            total_rounds=50,
            clients_per_round=10,
            adapter_type="lora",
            temperature=2.0
        )
        
        assert config.server_config.total_rounds == 50
        assert config.server_config.clients_per_round == 10
        assert config.client_config_template.adapter_type == "lora"
        assert config.distillation_config.temperature == 2.0


def test_end_to_end_simple():
    """Simple end-to-end test."""
    device = torch.device("cpu")
    
    # Create simple data
    def create_loader():
        X = torch.randn(50, 10)
        y = torch.randint(0, 2, (50,))
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=10)
    
    # Create client loaders
    client_loaders = {
        f"client_{i}": {"train": create_loader(), "val": create_loader()}
        for i in range(3)
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create config
        config = create_personalized_fl_config(
            total_rounds=2,
            clients_per_round=2,
            local_epochs=1,
            fl_config={
                "output_dir": Path(tmpdir),
                "experiment_name": "test"
            }
        )
        
        # Note: Full orchestrator test would require more setup
        # This just validates config creation
        assert config.server_config.total_rounds == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
