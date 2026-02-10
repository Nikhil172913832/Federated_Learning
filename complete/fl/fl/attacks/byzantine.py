"""Malicious client simulator for Byzantine robustness testing."""

import torch
import numpy as np
from typing import Dict, Literal

from fl.logging_config import get_logger

logger = get_logger(__name__)


class MaliciousClient:
    """Simulate malicious client behavior."""

    def __init__(
        self,
        attack_type: Literal[
            "random", "sign_flip", "gaussian", "label_flip"
        ] = "random",
    ):
        """Initialize malicious client.

        Args:
            attack_type: Type of attack to perform
        """
        self.attack_type = attack_type

    def corrupt_gradients(
        self, state_dict: Dict[str, torch.Tensor], intensity: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Corrupt model gradients.

        Args:
            state_dict: Original state dict
            intensity: Attack intensity (0-1)

        Returns:
            Corrupted state dict
        """
        corrupted = {}

        for name, param in state_dict.items():
            if self.attack_type == "random":
                corrupted[name] = torch.randn_like(param) * intensity

            elif self.attack_type == "sign_flip":
                corrupted[name] = -param * intensity

            elif self.attack_type == "gaussian":
                noise = torch.randn_like(param) * param.std() * intensity
                corrupted[name] = param + noise

            elif self.attack_type == "label_flip":
                corrupted[name] = param * (1 + intensity)

            else:
                corrupted[name] = param

        return corrupted


class ByzantineSimulator:
    """Simulate Byzantine attacks in federated learning."""

    def __init__(self, malicious_ratio: float = 0.2):
        """Initialize simulator.

        Args:
            malicious_ratio: Fraction of malicious clients (0-1)
        """
        if not (0 <= malicious_ratio < 1):
            raise ValueError(
                f"malicious_ratio must be in [0, 1), got {malicious_ratio}"
            )

        self.malicious_ratio = malicious_ratio

    def simulate_round(
        self,
        client_updates: list[Dict[str, torch.Tensor]],
        attack_type: str = "random",
        intensity: float = 1.0,
    ) -> list[Dict[str, torch.Tensor]]:
        """Simulate Byzantine attack in a training round.

        Args:
            client_updates: List of client state dicts
            attack_type: Type of attack
            intensity: Attack intensity

        Returns:
            List of updates with some corrupted
        """
        num_clients = len(client_updates)
        num_malicious = int(num_clients * self.malicious_ratio)

        if num_malicious == 0:
            return client_updates

        malicious_indices = np.random.choice(
            num_clients, size=num_malicious, replace=False
        )

        attacker = MaliciousClient(attack_type)
        corrupted_updates = []

        for i, update in enumerate(client_updates):
            if i in malicious_indices:
                corrupted = attacker.corrupt_gradients(update, intensity)
                corrupted_updates.append(corrupted)
                logger.debug(f"Client {i} is malicious ({attack_type})")
            else:
                corrupted_updates.append(update)

        logger.info(
            f"Simulated {num_malicious}/{num_clients} malicious clients "
            f"({self.malicious_ratio * 100:.0f}%)"
        )

        return corrupted_updates

    def evaluate_robustness(
        self,
        aggregator,
        honest_updates: list[Dict[str, torch.Tensor]],
        attack_types: list[str] = ["random", "sign_flip", "gaussian"],
    ) -> Dict[str, float]:
        """Evaluate aggregator robustness.

        Args:
            aggregator: Aggregation algorithm
            honest_updates: List of honest client updates
            attack_types: List of attack types to test

        Returns:
            Dictionary of attack type to robustness score
        """
        results = {}

        honest_aggregate = aggregator.aggregate(honest_updates)

        for attack_type in attack_types:
            corrupted_updates = self.simulate_round(
                honest_updates, attack_type=attack_type, intensity=1.0
            )

            robust_aggregate = aggregator.aggregate(corrupted_updates)

            distance = self._compute_distance(honest_aggregate, robust_aggregate)
            results[attack_type] = distance

            logger.info(f"{attack_type}: distance = {distance:.4f}")

        return results

    def _compute_distance(
        self, dict1: Dict[str, torch.Tensor], dict2: Dict[str, torch.Tensor]
    ) -> float:
        """Compute L2 distance between two state dicts.

        Args:
            dict1: First state dict
            dict2: Second state dict

        Returns:
            L2 distance
        """
        total_dist = 0.0

        for name in dict1.keys():
            if name in dict2:
                dist = torch.norm(dict1[name] - dict2[name])
                total_dist += dist.item() ** 2

        return np.sqrt(total_dist)
