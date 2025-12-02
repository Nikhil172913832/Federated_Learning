"""Aggregation strategies for federated learning."""

from fl.strategies.fedavg import FedAvgStrategy
from fl.strategies.fedprox import FedProxStrategy
from fl.strategies.fednova import FedNovaStrategy
from fl.strategies.scaffold import ScaffoldStrategy
from fl.strategies.fedadam import FedAdamStrategy

__all__ = [
    "FedAvgStrategy",
    "FedProxStrategy",
    "FedNovaStrategy",
    "ScaffoldStrategy",
    "FedAdamStrategy",
]
