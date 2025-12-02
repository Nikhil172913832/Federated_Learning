"""Byzantine-robust aggregation for defending against malicious clients."""

import torch
from typing import Dict, List, Tuple
from collections import OrderedDict
import numpy as np


class ByzantineRobustAggregator:
    """Byzantine-robust aggregation strategies.
    
    Protects against malicious clients sending corrupted updates.
    Implements multiple defense mechanisms.
    """

    def __init__(self, strategy: str = "krum"):
        """Initialize Byzantine-robust aggregator.

        Args:
            strategy: Defense strategy ('krum', 'trimmed_mean', 'median', 'bulyan')
        """
        self.strategy = strategy

    def aggregate(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        num_samples: List[int],
        num_byzantine: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client parameters with Byzantine tolerance.

        Args:
            client_params: List of client parameters
            num_samples: Number of samples per client
            num_byzantine: Expected number of Byzantine clients

        Returns:
            Robust aggregated parameters
        """
        if self.strategy == "krum":
            return self._krum(client_params, num_byzantine)
        elif self.strategy == "trimmed_mean":
            return self._trimmed_mean(client_params, num_byzantine)
        elif self.strategy == "median":
            return self._coordinate_median(client_params)
        elif self.strategy == "bulyan":
            return self._bulyan(client_params, num_byzantine)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _krum(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        num_byzantine: int,
    ) -> Dict[str, torch.Tensor]:
        """Krum aggregation: select update closest to majority.

        Original paper: Blanchard et al., 2017
        "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"

        Args:
            client_params: List of client parameters
            num_byzantine: Number of Byzantine clients to tolerate

        Returns:
            Selected client parameters
        """
        n = len(client_params)
        if n <= 2 * num_byzantine:
            raise ValueError("Not enough honest clients for Krum")

        # Flatten all parameters for distance computation
        flattened = []
        for params in client_params:
            flat = torch.cat([p.flatten() for p in params.values()])
            flattened.append(flat)

        # Compute pairwise distances
        scores = []
        n_select = n - num_byzantine - 2

        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = torch.norm(flattened[i] - flattened[j]).item()
                    distances.append(dist)

            # Sum of n_select closest distances
            distances.sort()
            score = sum(distances[:n_select])
            scores.append(score)

        # Select client with minimum score
        best_idx = np.argmin(scores)
        return client_params[best_idx]

    def _trimmed_mean(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        num_byzantine: int,
    ) -> Dict[str, torch.Tensor]:
        """Trimmed mean: remove extreme values and average.

        Args:
            client_params: List of client parameters
            num_byzantine: Number of extreme values to trim from each side

        Returns:
            Trimmed mean parameters
        """
        aggregated = OrderedDict()
        param_names = client_params[0].keys()

        for name in param_names:
            # Stack parameters across clients
            stacked = torch.stack([params[name] for params in client_params])

            # Sort along client dimension
            sorted_params, _ = torch.sort(stacked, dim=0)

            # Trim extremes and compute mean
            if num_byzantine > 0:
                trimmed = sorted_params[num_byzantine:-num_byzantine]
            else:
                trimmed = sorted_params

            aggregated[name] = torch.mean(trimmed, dim=0)

        return aggregated

    def _coordinate_median(
        self,
        client_params: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Coordinate-wise median aggregation.

        More robust than mean but can be biased.

        Args:
            client_params: List of client parameters

        Returns:
            Median parameters
        """
        aggregated = OrderedDict()
        param_names = client_params[0].keys()

        for name in param_names:
            # Stack parameters across clients
            stacked = torch.stack([params[name] for params in client_params])

            # Compute median along client dimension
            aggregated[name] = torch.median(stacked, dim=0)[0]

        return aggregated

    def _bulyan(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        num_byzantine: int,
    ) -> Dict[str, torch.Tensor]:
        """Bulyan aggregation: multi-Krum followed by trimmed mean.

        Original paper: El Mhamdi et al., 2018
        "The Hidden Vulnerability of Distributed Learning in Byzantium"

        More robust than Krum alone.

        Args:
            client_params: List of client parameters
            num_byzantine: Number of Byzantine clients

        Returns:
            Bulyan aggregated parameters
        """
        n = len(client_params)
        theta = num_byzantine

        # Select multiple good updates using Krum
        selected_indices = self._multi_krum(client_params, num_byzantine, theta=theta)

        # Apply trimmed mean on selected updates
        selected_params = [client_params[i] for i in selected_indices]
        return self._trimmed_mean(selected_params, num_byzantine=theta // 2)

    def _multi_krum(
        self,
        client_params: List[Dict[str, torch.Tensor]],
        num_byzantine: int,
        theta: int,
    ) -> List[int]:
        """Select multiple clients using Krum scoring.

        Args:
            client_params: List of client parameters
            num_byzantine: Number of Byzantine clients
            theta: Number of clients to select

        Returns:
            Indices of selected clients
        """
        n = len(client_params)
        n_select = n - num_byzantine - 2

        # Flatten parameters
        flattened = []
        for params in client_params:
            flat = torch.cat([p.flatten() for p in params.values()])
            flattened.append(flat)

        # Compute Krum scores
        scores = []
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    dist = torch.norm(flattened[i] - flattened[j]).item()
                    distances.append(dist)

            distances.sort()
            score = sum(distances[:n_select])
            scores.append(score)

        # Select theta clients with lowest scores
        sorted_indices = np.argsort(scores)
        return sorted_indices[:theta].tolist()


class AnomalyDetector:
    """Detect anomalous client updates."""

    def __init__(self, threshold: float = 3.0):
        """Initialize anomaly detector.

        Args:
            threshold: Z-score threshold for anomaly detection
        """
        self.threshold = threshold

    def detect_anomalies(
        self,
        client_params: List[Dict[str, torch.Tensor]],
    ) -> List[bool]:
        """Detect anomalous clients based on update norms.

        Args:
            client_params: List of client parameters

        Returns:
            List of boolean flags (True = anomalous)
        """
        # Compute L2 norm of each client's update
        norms = []
        for params in client_params:
            norm = sum(torch.norm(p).item() ** 2 for p in params.values()) ** 0.5
            norms.append(norm)

        # Compute z-scores
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)

        anomalies = []
        for norm in norms:
            if std_norm > 0:
                z_score = abs(norm - mean_norm) / std_norm
                anomalies.append(z_score > self.threshold)
            else:
                anomalies.append(False)

        return anomalies
