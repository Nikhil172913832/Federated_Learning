"""Simple CNN model for image classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from fl.models.base import BaseModel


class SimpleCNN(BaseModel):
    """Simple CNN adapted from PyTorch tutorial for MNIST/Medical imaging."""

    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        """Initialize SimpleCNN.

        Args:
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            num_classes: Number of output classes
        """
        super(SimpleCNN, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 28, 28)
            dummy = self.pool(F.relu(self.conv1(dummy)))
            dummy = self.pool(F.relu(self.conv2(dummy)))
            flattened_size = dummy.numel()

        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_input_shape(self) -> Tuple[int, int, int]:
        """Get expected input shape.

        Returns:
            Tuple of (channels, height, width)
        """
        return (self.input_channels, 28, 28)

    def get_output_size(self) -> int:
        """Get number of output classes.

        Returns:
            Number of classes
        """
        return self.num_classes


class ResidualBlock(nn.Module):
    """Residual block for deeper CNNs."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetFL(BaseModel):
    """Simplified ResNet for federated learning."""

    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        super(ResNetFL, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_input_shape(self) -> Tuple[int, int, int]:
        return (self.input_channels, 28, 28)

    def get_output_size(self) -> int:
        return self.num_classes
