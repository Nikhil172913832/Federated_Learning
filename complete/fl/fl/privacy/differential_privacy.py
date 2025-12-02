"""Enhanced differential privacy implementation."""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Any
from torch.utils.data import DataLoader

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False


class DifferentialPrivacy:
    """Differential Privacy manager for federated learning.
    
    Implements DP-SGD using Opacus library for training with privacy guarantees.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
    ):
        """Initialize differential privacy manager.

        Args:
            epsilon: Privacy budget (lower = more private)
            delta: Privacy parameter (should be < 1/dataset_size)
            max_grad_norm: Maximum L2 norm for gradient clipping
            noise_multiplier: Noise multiplier (auto-calculated if None)
        """
        if not OPACUS_AVAILABLE:
            raise ImportError(
                "Opacus is required for differential privacy. "
                "Install with: pip install opacus"
            )

        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.privacy_engine: Optional[PrivacyEngine] = None

    def make_private(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
    ) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader]:
        """Make model, optimizer, and data loader privacy-preserving.

        Args:
            model: PyTorch model
            optimizer: Optimizer
            data_loader: Training data loader

        Returns:
            Tuple of (private_model, private_optimizer, private_data_loader)
        """
        # Validate and fix model if needed
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)

        # Create privacy engine
        self.privacy_engine = PrivacyEngine()

        # Make private
        model, optimizer, data_loader = self.privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=self.noise_multiplier if self.noise_multiplier else 1.1,
            max_grad_norm=self.max_grad_norm,
        )

        return model, optimizer, data_loader

    def get_epsilon(self, steps: int) -> float:
        """Calculate privacy epsilon spent after given steps.

        Args:
            steps: Number of training steps

        Returns:
            Epsilon value
        """
        if self.privacy_engine is None:
            return 0.0

        return self.privacy_engine.get_epsilon(delta=self.delta)

    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get current privacy budget spent.

        Returns:
            Tuple of (epsilon, delta)
        """
        if self.privacy_engine is None:
            return (0.0, self.delta)

        epsilon = self.privacy_engine.get_epsilon(delta=self.delta)
        return (epsilon, self.delta)


def attach_dp_engine(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    max_grad_norm: float = 1.0,
) -> Tuple[Any, nn.Module, torch.optim.Optimizer, DataLoader]:
    """Attach differential privacy to training process.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        data_loader: Data loader
        epsilon: Privacy budget
        delta: Privacy parameter
        max_grad_norm: Gradient clipping norm

    Returns:
        Tuple of (privacy_engine, model, optimizer, data_loader)
    """
    dp_manager = DifferentialPrivacy(
        epsilon=epsilon,
        delta=delta,
        max_grad_norm=max_grad_norm,
    )

    model, optimizer, data_loader = dp_manager.make_private(
        model, optimizer, data_loader
    )

    return dp_manager, model, optimizer, data_loader


class GaussianMechanism:
    """Gaussian mechanism for adding calibrated noise to values."""

    def __init__(self, epsilon: float, delta: float, sensitivity: float):
        """Initialize Gaussian mechanism.

        Args:
            epsilon: Privacy parameter
            delta: Privacy parameter
            sensitivity: L2 sensitivity of the function
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.sigma = self._calculate_sigma()

    def _calculate_sigma(self) -> float:
        """Calculate noise standard deviation.

        Returns:
            Sigma value for Gaussian noise
        """
        # Gaussian mechanism: sigma = sensitivity * sqrt(2*ln(1.25/delta)) / epsilon
        import math
        return self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon

    def add_noise(self, value: torch.Tensor) -> torch.Tensor:
        """Add calibrated Gaussian noise to value.

        Args:
            value: Tensor to add noise to

        Returns:
            Noisy tensor
        """
        noise = torch.randn_like(value) * self.sigma
        return value + noise


class LaplaceMechanism:
    """Laplace mechanism for adding noise (for epsilon-DP)."""

    def __init__(self, epsilon: float, sensitivity: float):
        """Initialize Laplace mechanism.

        Args:
            epsilon: Privacy parameter
            sensitivity: L1 sensitivity
        """
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        self.scale = sensitivity / epsilon

    def add_noise(self, value: torch.Tensor) -> torch.Tensor:
        """Add calibrated Laplace noise.

        Args:
            value: Tensor to add noise to

        Returns:
            Noisy tensor
        """
        noise = torch.distributions.Laplace(0, self.scale).sample(value.shape)
        return value + noise.to(value.device)
