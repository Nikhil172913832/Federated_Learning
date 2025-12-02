"""Base model interface for federated learning."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple


class BaseModel(nn.Module, ABC):
    """Abstract base class for federated learning models."""

    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        pass

    @abstractmethod
    def get_input_shape(self) -> Tuple[int, ...]:
        """Get expected input shape (excluding batch dimension).

        Returns:
            Tuple representing input shape
        """
        pass

    @abstractmethod
    def get_output_size(self) -> int:
        """Get output size (number of classes).

        Returns:
            Number of output units
        """
        pass

    def count_parameters(self) -> int:
        """Count trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
