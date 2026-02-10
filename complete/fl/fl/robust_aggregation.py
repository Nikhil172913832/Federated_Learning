"""Byzantine-robust aggregation algorithms."""

import torch
from typing import List, Dict, Tuple
import numpy as np


class MultiKrumAggregator:
    """Multi-Krum Byzantine-robust aggregation."""

    def __init__(self, num_byzantine: int = 0, multi_k: int = 1):
        """Initialize Multi-Krum aggregator.

        Args:
            num_byzantine: Expected number of Byzantine clients
            multi_k: Number of closest gradients to average
        """
        self.num_byzantine = num_byzantine
        self.multi_k = multi_k

    def aggregate(
        self, client_updates: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client updates using Multi-Krum.

        Args:
            client_updates: List of client state dicts

        Returns:
            Aggregated state dict
        """
        if len(client_updates) == 0:
            raise ValueError("No client updates to aggregate")

        if len(client_updates) <= 2 * self.num_byzantine:
            raise ValueError(
                f"Need more than 2*num_byzantine clients, "
                f"got {len(client_updates)} clients and {self.num_byzantine} Byzantine"
            )

        n = len(client_updates)
        m = n - self.num_byzantine - 2

        param_names = list(client_updates[0].keys())

        flattened = []
        for update in client_updates:
            flat = torch.cat([update[name].flatten() for name in param_names])
            flattened.append(flat)

        scores = []
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = torch.norm(flattened[i] - flattened[j]).item()
                    distances.append(dist)
            distances.sort()
            score = sum(distances[:m])
            scores.append(score)

        top_k_indices = np.argsort(scores)[: self.multi_k]

        aggregated = {}
        for name in param_names:
            tensors = [client_updates[i][name] for i in top_k_indices]
            aggregated[name] = torch.mean(torch.stack(tensors), dim=0)

        return aggregated


class TrimmedMeanAggregator:
    """Trimmed Mean Byzantine-robust aggregation."""

    def __init__(self, trim_ratio: float = 0.1):
        """Initialize Trimmed Mean aggregator.

        Args:
            trim_ratio: Fraction of extreme values to trim from each side
        """
        if not (0 <= trim_ratio < 0.5):
            raise ValueError(f"trim_ratio must be in [0, 0.5), got {trim_ratio}")
        self.trim_ratio = trim_ratio

    def aggregate(
        self, client_updates: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client updates using Trimmed Mean.

        Args:
            client_updates: List of client state dicts

        Returns:
            Aggregated state dict
        """
        if len(client_updates) == 0:
            raise ValueError("No client updates to aggregate")

        n = len(client_updates)
        trim_count = int(n * self.trim_ratio)

        if trim_count * 2 >= n:
            raise ValueError(f"Trimming {trim_count} from each side leaves no values")

        param_names = list(client_updates[0].keys())
        aggregated = {}

        for name in param_names:
            stacked = torch.stack([update[name] for update in client_updates])

            sorted_vals, _ = torch.sort(stacked, dim=0)

            if trim_count > 0:
                trimmed = sorted_vals[trim_count:-trim_count]
            else:
                trimmed = sorted_vals

            aggregated[name] = torch.mean(trimmed, dim=0)

        return aggregated


class RobustAggregator:
    """Ensemble of robust aggregation methods."""

    def __init__(
        self,
        method: str = "trimmed_mean",
        num_byzantine: int = 0,
        trim_ratio: float = 0.1,
    ):
        """Initialize robust aggregator.

        Args:
            method: Aggregation method ('krum', 'trimmed_mean', 'median')
            num_byzantine: Expected number of Byzantine clients
            trim_ratio: Trim ratio for trimmed mean
        """
        self.method = method

        if method == "krum":
            self.aggregator = MultiKrumAggregator(num_byzantine, multi_k=1)
        elif method == "multi_krum":
            k = (
                max(1, len(client_updates) - num_byzantine)
                if hasattr(self, "client_updates")
                else 3
            )
            self.aggregator = MultiKrumAggregator(num_byzantine, multi_k=k)
        elif method == "trimmed_mean":
            self.aggregator = TrimmedMeanAggregator(trim_ratio)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def aggregate(
        self, client_updates: List[Dict[str, torch.Tensor]], weights: List[float] = None
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client updates.

        Args:
            client_updates: List of client state dicts
            weights: Optional weights for each client

        Returns:
            Aggregated state dict
        """
        return self.aggregator.aggregate(client_updates)

    def detect_byzantine(
        self, client_updates: List[Dict[str, torch.Tensor]], threshold: float = 3.0
    ) -> List[int]:
        """Detect potential Byzantine clients.

        Args:
            client_updates: List of client state dicts
            threshold: Standard deviations for outlier detection

        Returns:
            List of suspected Byzantine client indices
        """
        if len(client_updates) < 3:
            return []

        param_names = list(client_updates[0].keys())

        flattened = []
        for update in client_updates:
            flat = torch.cat([update[name].flatten() for name in param_names])
            flattened.append(flat)

        norms = [torch.norm(f).item() for f in flattened]
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)

        if std_norm == 0:
            return []

        byzantine = []
        for i, norm in enumerate(norms):
            z_score = abs(norm - mean_norm) / std_norm
            if z_score > threshold:
                byzantine.append(i)

        return byzantine
