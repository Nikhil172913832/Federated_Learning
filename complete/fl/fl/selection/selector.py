"""Base client selector interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class ClientSelector(ABC):
    """Abstract base class for client selection strategies."""

    def __init__(self, num_clients: int, **kwargs):
        """Initialize client selector.

        Args:
            num_clients: Total number of clients
            **kwargs: Additional configuration
        """
        self.num_clients = num_clients
        self.config = kwargs
        self.selection_history: List[List[int]] = []

    @abstractmethod
    def select(
        self,
        num_select: int,
        client_info: Optional[Dict[int, Dict[str, Any]]] = None,
        **kwargs,
    ) -> List[int]:
        """Select clients for the current round.

        Args:
            num_select: Number of clients to select
            client_info: Optional information about clients (data sizes, losses, etc.)
            **kwargs: Additional arguments

        Returns:
            List of selected client IDs
        """
        pass

    def record_selection(self, selected_clients: List[int]) -> None:
        """Record selected clients in history.

        Args:
            selected_clients: List of selected client IDs
        """
        self.selection_history.append(selected_clients)

    def get_selection_frequency(self) -> Dict[int, int]:
        """Get selection frequency for each client.

        Returns:
            Dictionary mapping client ID to selection count
        """
        frequency = {i: 0 for i in range(self.num_clients)}
        for selection in self.selection_history:
            for client_id in selection:
                frequency[client_id] += 1
        return frequency

    def get_fairness_metrics(self) -> Dict[str, float]:
        """Calculate fairness metrics across client selections.

        Returns:
            Dictionary of fairness metrics
        """
        frequency = self.get_selection_frequency()
        counts = list(frequency.values())

        if not counts:
            return {"variance": 0.0, "gini": 0.0, "min_count": 0, "max_count": 0}

        # Variance in selection counts
        variance = np.var(counts)

        # Gini coefficient
        sorted_counts = sorted(counts)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (2 * np.sum((i + 1) * sorted_counts[i] for i in range(n))) / (n * cumsum[-1]) - (n + 1) / n if cumsum[-1] > 0 else 0

        return {
            "variance": float(variance),
            "gini": float(gini),
            "min_count": min(counts),
            "max_count": max(counts),
        }
