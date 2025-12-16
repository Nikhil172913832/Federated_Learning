"""Core FL package."""

from .server import FederatedServer
from .client import FederatedClient
from .datasets import load_mnist_federated
from .model import SimpleCNN

__all__ = [
    "FederatedServer",
    "FederatedClient",
    "load_mnist_federated",
    "SimpleCNN",
]
