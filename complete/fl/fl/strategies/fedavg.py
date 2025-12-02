"""FedAvg aggregation strategy."""

import torch
from typing import Dict, List
from collections import OrderedDict

from fl.strategies.base import BaseStrategy


class FedAvgStrategy(BaseStrategy):
    """Federated Averaging (FedAvg) strategy.
    
    Original paper: McMahan et al., 2017
    "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    
    Simply averages client model parameters weighted by the number of samples.
    """

    def __init__(self, **kwargs):
        """Initialize FedAvg strategy.
        
        Args:
            **kwargs: Additional configuration (unused for basic FedAvg)
        """
        super().__init__(**kwargs)

    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        num_samples: List[int],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client parameters using weighted averaging.

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

        # Weighted average
        aggregated = OrderedDict()
        param_names = client_params[0].keys()

        for name in param_names:
            aggregated[name] = sum(
                client_params[i][name] * weights[i]
                for i in range(len(client_params))
            )

        return aggregated
