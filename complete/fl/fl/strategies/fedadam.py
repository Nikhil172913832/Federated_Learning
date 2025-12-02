"""FedAdam and FedYogi aggregation strategies."""

import torch
from typing import Dict, List, Optional
from collections import OrderedDict

from fl.strategies.base import BaseStrategy


class FedAdamStrategy(BaseStrategy):
    """Federated Adam (FedAdam) strategy.
    
    Original paper: Reddi et al., 2021
    "Adaptive Federated Optimization"
    
    Applies adaptive optimization (Adam) on the server side for aggregation.
    Maintains server-side momentum and second moment estimates.
    """

    def __init__(
        self,
        server_lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.99,
        epsilon: float = 1e-8,
        **kwargs,
    ):
        """Initialize FedAdam strategy.
        
        Args:
            server_lr: Server learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Optional[Dict[str, torch.Tensor]] = None  # First moment
        self.v: Optional[Dict[str, torch.Tensor]] = None  # Second moment

    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        num_samples: List[int],
        global_params: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client parameters using server-side Adam optimization.

        Args:
            client_params: List of client model parameters
            num_samples: Number of samples each client trained on
            global_params: Current global model parameters (required)
            **kwargs: Additional arguments

        Returns:
            Updated global model parameters
        """
        if not client_params:
            raise ValueError("No client parameters to aggregate")

        if global_params is None:
            raise ValueError("global_params required for FedAdam")

        # Compute pseudo-gradient: average of client updates
        total_samples = sum(num_samples)
        weights = [n / total_samples for n in num_samples]

        # Average client parameters
        avg_params = OrderedDict()
        param_names = client_params[0].keys()

        for name in param_names:
            avg_params[name] = sum(
                client_params[i][name] * weights[i]
                for i in range(len(client_params))
            )

        # Compute pseudo-gradient: delta = avg_params - global_params
        pseudo_grad = OrderedDict()
        for name in param_names:
            pseudo_grad[name] = avg_params[name] - global_params[name]

        # Initialize moments if first round
        if self.m is None:
            self.m = OrderedDict()
            self.v = OrderedDict()
            for name in param_names:
                self.m[name] = torch.zeros_like(global_params[name])
                self.v[name] = torch.zeros_like(global_params[name])

        # Update moments and global parameters using Adam
        updated_params = OrderedDict()
        for name in param_names:
            # Update biased first moment estimate
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * pseudo_grad[name]
            
            # Update biased second moment estimate
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (pseudo_grad[name] ** 2)
            
            # Compute bias-corrected moments
            m_hat = self.m[name] / (1 - self.beta1 ** (self.round_num + 1))
            v_hat = self.v[name] / (1 - self.beta2 ** (self.round_num + 1))
            
            # Update parameters
            updated_params[name] = global_params[name] + self.server_lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)

        return updated_params


class FedYogiStrategy(BaseStrategy):
    """Federated Yogi (FedYogi) strategy.
    
    Original paper: Reddi et al., 2021
    "Adaptive Federated Optimization"
    
    Uses Yogi optimizer on server side. Similar to FedAdam but with
    different second moment update rule for better adaptivity.
    """

    def __init__(
        self,
        server_lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.99,
        epsilon: float = 1e-8,
        **kwargs,
    ):
        """Initialize FedYogi strategy.
        
        Args:
            server_lr: Server learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Optional[Dict[str, torch.Tensor]] = None  # First moment
        self.v: Optional[Dict[str, torch.Tensor]] = None  # Second moment

    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        num_samples: List[int],
        global_params: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client parameters using server-side Yogi optimization.

        Args:
            client_params: List of client model parameters
            num_samples: Number of samples each client trained on
            global_params: Current global model parameters (required)
            **kwargs: Additional arguments

        Returns:
            Updated global model parameters
        """
        if not client_params:
            raise ValueError("No client parameters to aggregate")

        if global_params is None:
            raise ValueError("global_params required for FedYogi")

        # Compute pseudo-gradient
        total_samples = sum(num_samples)
        weights = [n / total_samples for n in num_samples]

        avg_params = OrderedDict()
        param_names = client_params[0].keys()

        for name in param_names:
            avg_params[name] = sum(
                client_params[i][name] * weights[i]
                for i in range(len(client_params))
            )

        pseudo_grad = OrderedDict()
        for name in param_names:
            pseudo_grad[name] = avg_params[name] - global_params[name]

        # Initialize moments if first round
        if self.m is None:
            self.m = OrderedDict()
            self.v = OrderedDict()
            for name in param_names:
                self.m[name] = torch.zeros_like(global_params[name])
                self.v[name] = torch.zeros_like(global_params[name])

        # Update using Yogi
        updated_params = OrderedDict()
        for name in param_names:
            # Update first moment
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * pseudo_grad[name]
            
            # Yogi second moment update (different from Adam)
            self.v[name] = self.v[name] - (1 - self.beta2) * (pseudo_grad[name] ** 2) * torch.sign(
                self.v[name] - pseudo_grad[name] ** 2
            )
            
            # Update parameters
            updated_params[name] = global_params[name] + self.server_lr * self.m[name] / (torch.sqrt(self.v[name]) + self.epsilon)

        return updated_params
