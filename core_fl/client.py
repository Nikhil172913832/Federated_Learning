"""Core federated learning client implementation.

This module provides the client-side logic for federated learning,
including local training on private data.
"""

from typing import Dict, Tuple
import copy

import torch
import torch.nn as nn
import torch.optim as optim


class FederatedClient:
    """Client for federated learning that trains on local data."""
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize federated client.
        
        Args:
            client_id: Unique identifier for this client
            model: Local model (copy of global model)
            train_loader: DataLoader for client's private training data
            device: Device to train on
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.dataset_size = len(train_loader.dataset)
        
    def set_weights(self, weights: Dict[str, torch.Tensor]):
        """Set local model weights from global model.
        
        Args:
            weights: Model state dict from server
        """
        self.model.load_state_dict(copy.deepcopy(weights))
    
    def train(
        self,
        epochs: int = 1,
        lr: float = 0.01,
        global_weights: Dict[str, torch.Tensor] = None,
        mu: float = 0.0,
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """Train local model on private data.
        
        Args:
            epochs: Number of local training epochs
            lr: Learning rate
            global_weights: Global model weights (for FedProx)
            mu: FedProx proximal term coefficient
            
        Returns:
            Tuple of (updated weights, average loss)
        """
        self.model.to(self.device)
        self.model.train()
        
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(epochs):
            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # FedProx: Add proximal term
                if mu > 0 and global_weights is not None:
                    proximal_term = 0.0
                    for name, param in self.model.named_parameters():
                        proximal_term += ((param - global_weights[name]) ** 2).sum()
                    loss += (mu / 2) * proximal_term
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return self.model.state_dict(), avg_loss
    
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate local model on local data.
        
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.to(self.device)
        self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return avg_loss, accuracy
