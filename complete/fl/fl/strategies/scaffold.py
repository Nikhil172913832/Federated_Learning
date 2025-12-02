"""SCAFFOLD aggregation strategy."""

import torch
from typing import Dict, List, Optional
from collections import OrderedDict

from fl.strategies.base import BaseStrategy


class ScaffoldStrategy(BaseStrategy):
    """SCAFFOLD (Stochastic Controlled Averaging for Federated Learning) strategy.
    
    Original paper: Karimireddy et al., 2020
    "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
    
    Uses control variates to reduce client drift in heterogeneous settings.
    Maintains server and client control variates for variance reduction.
    """

    def __init__(self, **kwargs):
        """Initialize SCAFFOLD strategy.
        
        Args:
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.server_controls: Optional[Dict[str, torch.Tensor]] = None
        self.client_controls: Dict[int, Dict[str, torch.Tensor]] = {}

    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        num_samples: List[int],
        client_controls: Optional[List[Dict[str, torch.Tensor]]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client parameters using SCAFFOLD.

        Args:
            client_params: List of client model parameters
            num_samples: Number of samples each client trained on
            client_controls: List of updated client control variates
            **kwargs: Additional arguments

        Returns:
            Aggregated model parameters
        """
        if not client_params:
            raise ValueError("No client parameters to aggregate")

        # Standard weighted average
        total_samples = sum(num_samples)
        weights = [n / total_samples for n in num_samples]

        aggregated = OrderedDict()
        param_names = client_params[0].keys()

        for name in param_names:
            aggregated[name] = sum(
                client_params[i][name] * weights[i]
                for i in range(len(client_params))
            )

        # Update server control variate
        if client_controls is not None:
            self._update_server_controls(client_controls, weights)

        return aggregated

    def _update_server_controls(
        self,
        client_controls: List[Dict[str, torch.Tensor]],
        weights: List[float],
    ) -> None:
        """Update server control variates.

        Args:
            client_controls: List of client control variates
            weights: Aggregation weights
        """
        if self.server_controls is None:
            # Initialize server controls
            self.server_controls = OrderedDict()
            for name in client_controls[0].keys():
                self.server_controls[name] = torch.zeros_like(client_controls[0][name])

        # Update: c_server = sum(weight_i * c_client_i)
        param_names = client_controls[0].keys()
        for name in param_names:
            self.server_controls[name] = sum(
                client_controls[i][name] * weights[i]
                for i in range(len(client_controls))
            )

    def get_server_controls(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get current server control variates.
        
        Returns:
            Server control variates or None if not initialized
        """
        return self.server_controls

    def initialize_client_controls(
        self,
        client_id: int,
        model_params: Dict[str, torch.Tensor],
    ) -> None:
        """Initialize control variates for a new client.

        Args:
            client_id: Client identifier
            model_params: Model parameters to initialize controls from
        """
        self.client_controls[client_id] = OrderedDict()
        for name, param in model_params.items():
            self.client_controls[client_id][name] = torch.zeros_like(param)
