"""FedProx aggregation strategy."""

import torch
from typing import Dict, List
from collections import OrderedDict

from fl.strategies.base import BaseStrategy


class FedProxStrategy(BaseStrategy):
    """Federated Proximal (FedProx) strategy.
    
    Original paper: Li et al., 2020
    "Federated Optimization in Heterogeneous Networks"
    
    Adds a proximal term during local training to handle system heterogeneity.
    The aggregation is the same as FedAvg, but clients use a proximal term
    during training to stay close to the global model.
    """

    def __init__(self, mu: float = 0.01, **kwargs):
        """Initialize FedProx strategy.
        
        Args:
            mu: Proximal term coefficient (controls similarity to global model)
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.mu = mu

    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        num_samples: List[int],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client parameters using weighted averaging.

        Note: The proximal regularization happens during client training,
        not during aggregation. Server-side aggregation is identical to FedAvg.

        Args:
            client_params: List of client model parameters
            num_samples: Number of samples each client trained on
            **kwargs: Unused additional arguments

        Returns:
            Aggregated model parameters
        """
        if not client_params:
            raise ValueError("No client parameters to aggregate")

        if len(client_params) != len(num_samples):
            raise ValueError("Mismatch between client_params and num_samples lengths")

        # Calculate weights based on number of samples
        total_samples = sum(num_samples)
        weights = [n / total_samples for n in num_samples]

        # Weighted average (same as FedAvg)
        aggregated = OrderedDict()
        param_names = client_params[0].keys()

        for name in param_names:
            aggregated[name] = sum(
                client_params[i][name] * weights[i]
                for i in range(len(client_params))
            )

        return aggregated

    def get_proximal_mu(self) -> float:
        """Get the proximal term coefficient for clients.
        
        Returns:
            Proximal mu value
        """
        return self.mu
