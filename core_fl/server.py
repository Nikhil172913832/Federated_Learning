"""Core federated learning server implementation.

This module provides the server-side logic for federated learning,
including model aggregation and coordination of training rounds.
"""

from typing import List, Dict, Any, Tuple
import copy

import torch
import torch.nn as nn


class FederatedServer:
    """Server for federated learning that aggregates client updates."""
    
    def __init__(self, model: nn.Module, strategy: str = "fedavg"):
        """Initialize federated server.
        
        Args:
            model: Global model to be trained
            strategy: Aggregation strategy ("fedavg" or "fedprox")
        """
        self.global_model = model
        self.strategy = strategy
        self.round_num = 0
        
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global model weights.
        
        Returns:
            Dictionary of model state dict
        """
        return copy.deepcopy(self.global_model.state_dict())
    
    def aggregate_weights(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client model weights using FedAvg.
        
        Args:
            client_weights: List of client model state dicts
            client_sizes: List of dataset sizes for each client
            
        Returns:
            Aggregated model state dict
            
        Raises:
            ValueError: If client_weights is empty or lengths don't match
        """
        # Input validation
        if not client_weights:
            raise ValueError("client_weights cannot be empty")
        if len(client_weights) != len(client_sizes):
            raise ValueError(
                f"Length mismatch: {len(client_weights)} weights vs {len(client_sizes)} sizes"
            )
        if any(size <= 0 for size in client_sizes):
            raise ValueError("All client_sizes must be positive")
        
        if self.strategy == "fedavg":
            return self._fedavg_aggregate(client_weights, client_sizes)
        elif self.strategy == "fedprox":
            # FedProx uses same aggregation as FedAvg
            return self._fedavg_aggregate(client_weights, client_sizes)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _fedavg_aggregate(
        self,
        client_weights: List[Dict[str, torch.Tensor]],
        client_sizes: List[int],
    ) -> Dict[str, torch.Tensor]:
        """FedAvg: Weighted average of client models.
        
        Args:
            client_weights: List of client model state dicts
            client_sizes: List of dataset sizes for each client
            
        Returns:
            Aggregated model state dict
        """
        total_size = sum(client_sizes)
        aggregated_weights = {}
        
        # Get all parameter names from first client
        param_names = client_weights[0].keys()
        
        for param_name in param_names:
            # Weighted sum of parameters
            aggregated_weights[param_name] = sum(
                client_weights[i][param_name] * (client_sizes[i] / total_size)
                for i in range(len(client_weights))
            )
        
        return aggregated_weights
    
    def update_global_model(self, aggregated_weights: Dict[str, torch.Tensor]):
        """Update global model with aggregated weights.
        
        Args:
            aggregated_weights: Aggregated model state dict
        """
        self.global_model.load_state_dict(aggregated_weights)
        self.round_num += 1
    
    def evaluate(
        self,
        test_loader,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[float, float]:
        """Evaluate global model on test data.
        
        Args:
            test_loader: DataLoader for test data
            device: Device to run evaluation on
            
        Returns:
            Tuple of (loss, accuracy)
        """
        self.global_model.to(device)
        self.global_model.eval()
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                
                outputs = self.global_model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
