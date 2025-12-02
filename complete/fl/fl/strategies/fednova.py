"""FedNova aggregation strategy."""

import torch
from typing import Dict, List, Optional
from collections import OrderedDict

from fl.strategies.base import BaseStrategy


class FedNovaStrategy(BaseStrategy):
    """Federated Normalized Averaging (FedNova) strategy.
    
    Original paper: Wang et al., 2020
    "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization"
    
    Normalizes local updates to account for heterogeneous local training.
    Handles varying numbers of local steps across clients.
    """

    def __init__(self, **kwargs):
        """Initialize FedNova strategy.
        
        Args:
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)

    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        num_samples: List[int],
        local_steps: Optional[List[int]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client parameters using normalized averaging.

        Args:
            client_params: List of client model parameters
            num_samples: Number of samples each client trained on
            local_steps: Number of local training steps per client (required)
            **kwargs: Additional arguments

        Returns:
            Aggregated model parameters
        """
        if not client_params:
            raise ValueError("No client parameters to aggregate")

        if local_steps is None:
            # Default to equal steps if not provided
            local_steps = [1] * len(client_params)

        if len(client_params) != len(num_samples) != len(local_steps):
            raise ValueError("Mismatch in lengths of client_params, num_samples, and local_steps")

        # Calculate effective number of steps (tau_i in the paper)
        # tau_i = local_steps[i] is the number of local SGD steps
        total_samples = sum(num_samples)

        # Normalized weights
        # Each client's contribution is weighted by samples and normalized by local steps
        normalized_weights = []
        for i in range(len(client_params)):
            weight = num_samples[i] / total_samples
            normalized_weight = weight * local_steps[i]
            normalized_weights.append(normalized_weight)

        # Normalize weights to sum to 1
        weight_sum = sum(normalized_weights)
        normalized_weights = [w / weight_sum for w in normalized_weights]

        # Weighted average with normalized weights
        aggregated = OrderedDict()
        param_names = client_params[0].keys()

        for name in param_names:
            aggregated[name] = sum(
                client_params[i][name] * normalized_weights[i]
                for i in range(len(client_params))
            )

        return aggregated
