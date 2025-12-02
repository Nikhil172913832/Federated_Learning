"""Core federated learning components."""

from fl.core.client import FederatedClient
from fl.core.server import FederatedServer

__all__ = ["FederatedClient", "FederatedServer"]
