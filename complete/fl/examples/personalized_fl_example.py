"""
Personalized Federated Learning with Knowledge Distillation example.

Demonstrates personalized FL with LoRA adapters and heterogeneous client data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from pathlib import Path

from fl.personalization import (
    create_personalized_fl_config,
    PersonalizedFLOrchestrator,
)
from fl.models.cnn import SimpleCNN


def create_heterogeneous_data_loaders(
    num_clients: int = 20,
    alpha: float = 0.5,
    batch_size: int = 32
):
    """Create heterogeneous data loaders using Dirichlet distribution."""
    print(f"Creating heterogeneous data for {num_clients} clients (alpha={alpha})...")
    
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Partition training data with Dirichlet distribution
    num_classes = 10
    labels = torch.tensor([y for _, y in train_dataset])
    
    client_data_indices = {i: [] for i in range(num_clients)}
    
    for k in range(num_classes):
        # Get indices for class k
        idx_k = (labels == k).nonzero(as_tuple=True)[0].numpy()
        
        # Sample from Dirichlet
        proportions = torch.distributions.Dirichlet(
            torch.tensor([alpha] * num_clients)
        ).sample()
        
        # Assign data to clients
        proportions = (proportions * len(idx_k)).int().numpy()
        
        start_idx = 0
        for client_id, proportion in enumerate(proportions):
            end_idx = start_idx + proportion
            client_data_indices[client_id].extend(idx_k[start_idx:end_idx].tolist())
            start_idx = end_idx
            
    # Create data loaders for each client
    client_loaders = {}
    
    for client_id in range(num_clients):
        indices = client_data_indices[client_id]
        
        if len(indices) == 0:
            continue
            
        # Split into train/val
        val_size = max(10, int(0.1 * len(indices)))
        train_indices = indices[:-val_size]
        val_indices = indices[-val_size:]
        
        # Create datasets
        train_subset = torch.utils.data.Subset(train_dataset, train_indices)
        val_subset = torch.utils.data.Subset(train_dataset, val_indices)
        
        # Create loaders
        client_loaders[f"client_{client_id}"] = {
            "train": DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2
            ),
            "val": DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2
            ),
            "test": DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2
            )
        }
        
        print(f"  Client {client_id}: {len(train_indices)} train, {len(val_indices)} val samples")
        
    # Global test loader
    global_test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return client_loaders, global_test_loader


def create_model():
    """Create a fresh model instance."""
    return SimpleCNN(num_classes=10)


def main():
    """Run personalized FL experiment."""
    print("=" * 80)
    print("Personalized Federated Learning with Knowledge Distillation")
    print("=" * 80)
    
    num_clients = 20
    clients_per_round = 10
    total_rounds = 100
    
    client_loaders, global_test_loader = create_heterogeneous_data_loaders(
        num_clients=num_clients,
        alpha=0.5,
        batch_size=32
    )
    
    config = create_personalized_fl_config(
        total_rounds=total_rounds,
        clients_per_round=clients_per_round,
        local_epochs=1,
        learning_rate=1e-3,
        adapter_type="lora",
        adapter_rank=4,
        temperature=2.0,
        lambda_kd=0.5,
        server_config={
            "aggregation_strategy": "fedavg",
            "weighted_aggregation": True,
            "enable_anomaly_detection": True,
            "checkpoint_every_n_rounds": 10,
            "eval_every_n_rounds": 5,
        },
        client_config={
            "batch_size": 32,
            "freeze_backbone": True,
            "clip_norm": 1.0,
            "min_samples_for_personalization": 50,
            "personalization_warmup_rounds": 5,
        },
        fl_config={
            "experiment_name": "personalized_fl_cifar10",
            "output_dir": Path("./outputs"),
            "use_progressive_distillation": True,
            "verbose": True,
        }
    )
    
    orchestrator = PersonalizedFLOrchestrator(
        config=config,
        model_fn=create_model,
        client_data_loaders=client_loaders,
        global_test_loader=global_test_loader
    )
    
    print("\nStarting training...")
    results = orchestrator.run()
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nFinal Results:")
    print(f"  Training time: {results['training_time']:.2f}s")
    
    if "global_test" in results["final_metrics"]:
        global_acc = results["final_metrics"]["global_test"]["global_accuracy"]
        print(f"  Global model accuracy: {global_acc:.4f}")
        
    if "client_test" in results["final_metrics"]:
        client_acc = results["final_metrics"]["client_test"].get("avg_accuracy", 0.0)
        print(f"  Average personalized accuracy: {client_acc:.4f}")
        
    if "personalization_gains" in results["final_metrics"]:
        gains = results["final_metrics"]["personalization_gains"]
        print(f"\nPersonalization Gains:")
        print(f"  Mean gain: {gains.get('mean_gain', 0.0):.4f}")
        print(f"  Median gain: {gains.get('median_gain', 0.0):.4f}")
        print(f"  Max gain: {gains.get('max_gain', 0.0):.4f}")
        print(f"  Min gain: {gains.get('min_gain', 0.0):.4f}")
        
    print(f"\nResults saved to: {config.output_dir / config.experiment_name}")


if __name__ == "__main__":
    main()
