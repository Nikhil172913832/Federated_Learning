"""Importance sampling-based client selection."""

import numpy as np
from typing import List, Dict, Any, Optional

from fl.selection.selector import ClientSelector


class ImportanceSamplingSelector(ClientSelector):
    """Select clients based on importance sampling.
    
    Clients with more data or higher loss are more likely to be selected.
    """

    def __init__(
        self,
        num_clients: int,
        sampling_method: str = "data_size",
        **kwargs,
    ):
        """Initialize importance sampling selector.

        Args:
            num_clients: Total number of clients
            sampling_method: Method for computing importance ('data_size', 'loss', 'gradient_norm')
            **kwargs: Additional configuration
        """
        super().__init__(num_clients, **kwargs)
        self.sampling_method = sampling_method

    def select(
        self,
        num_select: int,
        client_info: Optional[Dict[int, Dict[str, Any]]] = None,
        **kwargs,
    ) -> List[int]:
        """Select clients using importance sampling.

        Args:
            num_select: Number of clients to select
            client_info: Dictionary with client information (data_size, loss, etc.)
            **kwargs: Additional arguments

        Returns:
            List of selected client IDs
        """
        if client_info is None:
            # Fall back to uniform sampling
            return np.random.choice(
                self.num_clients,
                size=num_select,
                replace=False,
            ).tolist()

        # Compute importance scores
        scores = self._compute_importance_scores(client_info)

        # Normalize to probabilities
        if sum(scores) == 0:
            probabilities = np.ones(self.num_clients) / self.num_clients
        else:
            probabilities = scores / sum(scores)

        # Sample based on probabilities
        selected = np.random.choice(
            self.num_clients,
            size=num_select,
            replace=False,
            p=probabilities,
        ).tolist()

        self.record_selection(selected)
        return selected

    def _compute_importance_scores(
        self,
        client_info: Dict[int, Dict[str, Any]],
    ) -> np.ndarray:
        """Compute importance scores for all clients.

        Args:
            client_info: Client information dictionary

        Returns:
            Array of importance scores
        """
        scores = np.zeros(self.num_clients)

        for client_id in range(self.num_clients):
            if client_id not in client_info:
                scores[client_id] = 1.0  # Default score
                continue

            info = client_info[client_id]

            if self.sampling_method == "data_size":
                # More data = higher importance
                scores[client_id] = info.get("data_size", 1.0)

            elif self.sampling_method == "loss":
                # Higher loss = higher importance
                scores[client_id] = info.get("loss", 1.0)

            elif self.sampling_method == "gradient_norm":
                # Larger gradient = higher importance
                scores[client_id] = info.get("gradient_norm", 1.0)

            else:
                scores[client_id] = 1.0

        return scores


class AdaptiveImportanceSelector(ClientSelector):
    """Adaptive importance sampling with staleness consideration."""

    def __init__(self, num_clients: int, staleness_weight: float = 0.3, **kwargs):
        """Initialize adaptive importance selector.

        Args:
            num_clients: Total number of clients
            staleness_weight: Weight for staleness factor (0 to 1)
            **kwargs: Additional configuration
        """
        super().__init__(num_clients, **kwargs)
        self.staleness_weight = staleness_weight
        self.last_selected_round = {i: -1 for i in range(num_clients)}
        self.current_round = 0

    def select(
        self,
        num_select: int,
        client_info: Optional[Dict[int, Dict[str, Any]]] = None,
        **kwargs,
    ) -> List[int]:
        """Select clients with adaptive importance considering staleness.

        Args:
            num_select: Number of clients to select
            client_info: Client information dictionary
            **kwargs: Additional arguments

        Returns:
            List of selected client IDs
        """
        # Compute base importance scores
        if client_info is not None:
            base_scores = np.array([
                client_info.get(i, {}).get("data_size", 1.0)
                for i in range(self.num_clients)
            ])
        else:
            base_scores = np.ones(self.num_clients)

        # Compute staleness scores (higher = more stale = higher priority)
        staleness_scores = np.array([
            self.current_round - self.last_selected_round[i]
            for i in range(self.num_clients)
        ])

        # Combine scores
        combined_scores = (
            (1 - self.staleness_weight) * base_scores +
            self.staleness_weight * staleness_scores
        )

        # Normalize to probabilities
        if sum(combined_scores) == 0:
            probabilities = np.ones(self.num_clients) / self.num_clients
        else:
            probabilities = combined_scores / sum(combined_scores)

        # Sample
        selected = np.random.choice(
            self.num_clients,
            size=num_select,
            replace=False,
            p=probabilities,
        ).tolist()

        # Update tracking
        for client_id in selected:
            self.last_selected_round[client_id] = self.current_round

        self.current_round += 1
        self.record_selection(selected)
        return selected
