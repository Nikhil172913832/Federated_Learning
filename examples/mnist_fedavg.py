#!/usr/bin/env python3
"""
Federated Learning Demo: MNIST with FedAvg

This script demonstrates federated learning in action:
- 10 clients train on private MNIST data
- Server aggregates using FedAvg
- Runs for 10 rounds
- Prints accuracy improvement

Expected runtime: < 2 minutes on CPU
Expected final accuracy: ~97%
"""

import torch
import sys
from pathlib import Path

# Add core_fl to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_fl import FederatedServer, FederatedClient, load_mnist_federated, SimpleCNN


def main():
    print("=" * 60)
    print("FEDERATED LEARNING DEMO: MNIST with FedAvg")
    print("=" * 60)
    print()
    
    # Configuration
    NUM_CLIENTS = 10
    NUM_ROUNDS = 10
    LOCAL_EPOCHS = 1
    LEARNING_RATE = 0.01
    BATCH_SIZE = 32
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Configuration:")
    print(f"  Clients: {NUM_CLIENTS}")
    print(f"  Rounds: {NUM_ROUNDS}")
    print(f"  Local epochs: {LOCAL_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Device: {DEVICE}")
    print()
    
    # Load data
    print("Loading MNIST dataset...")
    client_loaders, test_loader = load_mnist_federated(
        num_clients=NUM_CLIENTS,
        batch_size=BATCH_SIZE,
        iid=True,
    )
    print(f"✓ Loaded data for {len(client_loaders)} clients")
    print(f"✓ Test set: {len(test_loader.dataset)} samples")
    print()
    
    # Initialize global model
    global_model = SimpleCNN(num_classes=10)
    
    # Initialize server
    server = FederatedServer(global_model, strategy="fedavg")
    
    # Initialize clients
    clients = []
    for i, loader in enumerate(client_loaders):
        client_model = SimpleCNN(num_classes=10)
        client = FederatedClient(
            client_id=i,
            model=client_model,
            train_loader=loader,
            device=DEVICE,
        )
        clients.append(client)
    
    print(f"✓ Initialized server and {len(clients)} clients")
    print()
    
    # Federated learning rounds
    print("Starting federated learning...")
    print("-" * 60)
    print(f"{'Round':<8} {'Train Loss':<12} {'Test Acc':<12} {'Test Loss':<12}")
    print("-" * 60)
    
    for round_num in range(NUM_ROUNDS):
        # Get global weights
        global_weights = server.get_global_weights()
        
        # Client training
        client_weights = []
        client_sizes = []
        total_loss = 0.0
        
        for client in clients:
            # Set global weights
            client.set_weights(global_weights)
            
            # Local training
            weights, loss = client.train(
                epochs=LOCAL_EPOCHS,
                lr=LEARNING_RATE,
            )
            
            client_weights.append(weights)
            client_sizes.append(client.dataset_size)
            total_loss += loss
        
        avg_train_loss = total_loss / len(clients)
        
        # Server aggregation
        aggregated_weights = server.aggregate_weights(client_weights, client_sizes)
        server.update_global_model(aggregated_weights)
        
        # Evaluate global model
        test_loss, test_acc = server.evaluate(test_loader, device=DEVICE)
        
        # Print progress
        print(f"{round_num + 1:<8} {avg_train_loss:<12.4f} {test_acc:<12.4f} {test_loss:<12.4f}")
    
    print("-" * 60)
    print()
    
    # Final results
    final_loss, final_acc = server.evaluate(test_loader, device=DEVICE)
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {final_acc * 100:.2f}%")
    print(f"Test Loss: {final_loss:.4f}")
    print()
    print("✓ Federated learning completed successfully!")
    print()
    
    # Show what happened
    print("What just happened:")
    print("  1. Loaded MNIST and split across 10 clients")
    print("  2. Each client trained locally on private data")
    print("  3. Server aggregated updates using FedAvg")
    print("  4. Repeated for 10 rounds")
    print(f"  5. Achieved {final_acc * 100:.1f}% accuracy without sharing raw data!")
    print()


if __name__ == "__main__":
    main()
