"""Federated Learning Server implementation."""

import torch
from typing import Dict, List, Any, Optional
from collections import OrderedDict


class FederatedServer:
    """Handles server-side aggregation and coordination in federated learning."""

    def __init__(
        self,
        model: torch.nn.Module,
        strategy: str = "fedavg",
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a federated learning server.

        Args:
            model: Global model to coordinate training for
            strategy: Aggregation strategy (fedavg, fedprox, etc.)
            config: Optional configuration dictionary
        """
        self.model = model
        self.strategy = strategy
        self.config = config or {}
        self.round_num = 0
        self.metrics_history: List[Dict[str, Any]] = []

    def aggregate_parameters(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        num_samples: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client parameters using the specified strategy.

        Args:
            client_params: List of client model parameters
            num_samples: Number of samples each client trained on

        Returns:
            Aggregated model parameters
        """
        if self.strategy == "fedavg":
            return self._fedavg_aggregate(client_params, num_samples)
        elif self.strategy == "fedprox":
            return self._fedavg_aggregate(client_params, num_samples)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _fedavg_aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        num_samples: List[int],
    ) -> Dict[str, torch.Tensor]:
        """FedAvg weighted averaging aggregation.

        Args:
            client_params: List of client model parameters
            num_samples: Number of samples each client trained on

        Returns:
            Aggregated model parameters
        """
        if not client_params:
            return self.model.state_dict()

        total_samples = sum(num_samples)
        aggregated_params = OrderedDict()

        # Get all parameter names from the first client
        param_names = client_params[0].keys()

        for name in param_names:
            # Weighted average of parameters
            aggregated_params[name] = sum(
                client_params[i][name] * (num_samples[i] / total_samples)
                for i in range(len(client_params))
            )

        return aggregated_params

    def update_global_model(self, aggregated_params: Dict[str, torch.Tensor]) -> None:
        """Update the global model with aggregated parameters.

        Args:
            aggregated_params: Aggregated parameters from clients
        """
        self.model.load_state_dict(aggregated_params)
        self.round_num += 1

    def get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current global model parameters."""
        return self.model.state_dict()

    def log_round_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics for the current round.

        Args:
            metrics: Dictionary of metrics to log
        """
        metrics["round"] = self.round_num
        self.metrics_history.append(metrics)

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get full metrics history."""
        return self.metrics_history
