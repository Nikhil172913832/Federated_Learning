"""Random client selection strategy."""

import numpy as np
from typing import List, Dict, Any, Optional

from fl.selection.selector import ClientSelector


class RandomSelector(ClientSelector):
    """Randomly select clients for each round.
    
    This is the baseline selection strategy used in FedAvg.
    """

    def __init__(self, num_clients: int, seed: Optional[int] = None, **kwargs):
        """Initialize random selector.

        Args:
            num_clients: Total number of clients
            seed: Random seed for reproducibility
            **kwargs: Additional configuration
        """
        super().__init__(num_clients, **kwargs)
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def select(
        self,
        num_select: int,
        client_info: Optional[Dict[int, Dict[str, Any]]] = None,
        **kwargs,
    ) -> List[int]:
        """Randomly select clients.

        Args:
            num_select: Number of clients to select
            client_info: Unused (included for interface consistency)
            **kwargs: Additional arguments

        Returns:
            List of selected client IDs
        """
        if num_select > self.num_clients:
            raise ValueError(f"Cannot select {num_select} clients from {self.num_clients}")

        # Random sampling without replacement
        selected = np.random.choice(
            self.num_clients,
            size=num_select,
            replace=False,
        ).tolist()

        self.record_selection(selected)
        return selected


class UniformRandomSelector(ClientSelector):
    """Uniform random selection ensuring all clients selected equally over time."""

    def __init__(self, num_clients: int, **kwargs):
        """Initialize uniform random selector.

        Args:
            num_clients: Total number of clients
            **kwargs: Additional configuration
        """
        super().__init__(num_clients, **kwargs)
        self.remaining_pool = list(range(num_clients))

    def select(
        self,
        num_select: int,
        client_info: Optional[Dict[int, Dict[str, Any]]] = None,
        **kwargs,
    ) -> List[int]:
        """Select clients uniformly, cycling through all clients.

        Args:
            num_select: Number of clients to select
            client_info: Unused
            **kwargs: Additional arguments

        Returns:
            List of selected client IDs
        """
        # Refill pool if exhausted
        if len(self.remaining_pool) < num_select:
            self.remaining_pool = list(range(self.num_clients))
            np.random.shuffle(self.remaining_pool)

        # Select from remaining pool
        selected = self.remaining_pool[:num_select]
        self.remaining_pool = self.remaining_pool[num_select:]

        self.record_selection(selected)
        return selected
