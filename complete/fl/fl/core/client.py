"""Federated Learning Client implementation."""

import torch
from typing import Dict, Any, Tuple, Optional
from torch.utils.data import DataLoader

from fl.utils.metrics import compute_accuracy


class FederatedClient:
    """Handles client-side training and evaluation in federated learning."""

    def __init__(
        self,
        client_id: int,
        model: torch.nn.Module,
        device: torch.device,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a federated learning client.

        Args:
            client_id: Unique identifier for this client
            model: PyTorch model to train
            device: Device to run computations on
            config: Optional configuration dictionary
        """
        self.client_id = client_id
        self.model = model
        self.device = device
        self.config = config or {}
        self.model.to(self.device)

    def train(
        self,
        trainloader: DataLoader,
        epochs: int,
        lr: float,
        global_params: Optional[Dict] = None,
    ) -> Tuple[Dict[str, torch.Tensor], float, int]:
        """Train the model on local data.

        Args:
            trainloader: DataLoader for training data
            epochs: Number of local training epochs
            lr: Learning rate
            global_params: Optional global model parameters for proximal term

        Returns:
            Tuple of (updated model parameters, average loss, number of samples)
        """
        self.model.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        running_loss = 0.0
        num_samples = 0

        for _ in range(epochs):
            for batch in trainloader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Add proximal term if using FedProx
                if global_params is not None:
                    mu = self.config.get("fedprox_mu", 0.01)
                    prox_term = 0.0
                    for name, param in self.model.named_parameters():
                        if name in global_params:
                            prox_term += ((param - global_params[name]) ** 2).sum()
                    loss += (mu / 2) * prox_term

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * len(images)
                num_samples += len(images)

        avg_loss = running_loss / num_samples if num_samples > 0 else 0.0
        return self.model.state_dict(), avg_loss, num_samples

    def evaluate(
        self, testloader: DataLoader
    ) -> Tuple[float, float, int]:
        """Evaluate the model on local data.

        Args:
            testloader: DataLoader for test data

        Returns:
            Tuple of (loss, accuracy, number of samples)
        """
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in testloader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * len(labels)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += len(labels)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy, total

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current model parameters."""
        return self.model.state_dict()

    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Set model parameters."""
        self.model.load_state_dict(parameters)
