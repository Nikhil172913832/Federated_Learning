"""Base strategy interface for federated learning aggregation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import torch
from collections import OrderedDict


class BaseStrategy(ABC):
    """Abstract base class for federated learning aggregation strategies."""

    def __init__(self, **kwargs):
        """Initialize the strategy.
        
        Args:
            **kwargs: Strategy-specific configuration parameters
        """
        self.config = kwargs
        self.round_num = 0

    @abstractmethod
    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        num_samples: List[int],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client parameters.

        Args:
            client_params: List of client model parameters
            num_samples: Number of samples each client trained on
            **kwargs: Additional strategy-specific arguments

        Returns:
            Aggregated model parameters
        """
        pass

    def on_round_begin(self, round_num: int) -> None:
        """Hook called at the beginning of each round.

        Args:
            round_num: Current round number
        """
        self.round_num = round_num

    def on_round_end(self, global_params: Dict[str, torch.Tensor]) -> None:
        """Hook called at the end of each round.

        Args:
            global_params: Aggregated global model parameters
        """
        pass

    @staticmethod
    def weighted_average(
        params_list: List[Dict[str, torch.Tensor]],
        weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted average of parameters.

        Args:
            params_list: List of parameter dictionaries
            weights: List of weights (should sum to 1.0)

        Returns:
            Weighted average of parameters
        """
        if not params_list:
            return {}

        result = OrderedDict()
        param_names = params_list[0].keys()

        for name in param_names:
            result[name] = sum(
                params[name] * weight
                for params, weight in zip(params_list, weights)
            )

        return result
