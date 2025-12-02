"""Fairness-aware client selection strategies."""

import numpy as np
from typing import List, Dict, Any, Optional

from fl.selection.selector import ClientSelector


class FairnessAwareSelector(ClientSelector):
    """Select clients to ensure fairness in participation.
    
    Ensures all clients get equal opportunity to participate over time.
    """

    def __init__(
        self,
        num_clients: int,
        fairness_criterion: str = "equal_participation",
        **kwargs,
    ):
        """Initialize fairness-aware selector.

        Args:
            num_clients: Total number of clients
            fairness_criterion: Fairness criterion ('equal_participation', 'min_variance')
            **kwargs: Additional configuration
        """
        super().__init__(num_clients, **kwargs)
        self.fairness_criterion = fairness_criterion
        self.selection_counts = np.zeros(num_clients)

    def select(
        self,
        num_select: int,
        client_info: Optional[Dict[int, Dict[str, Any]]] = None,
        **kwargs,
    ) -> List[int]:
        """Select clients with fairness consideration.

        Args:
            num_select: Number of clients to select
            client_info: Optional client information
            **kwargs: Additional arguments

        Returns:
            List of selected client IDs
        """
        if self.fairness_criterion == "equal_participation":
            selected = self._equal_participation_selection(num_select)
        elif self.fairness_criterion == "min_variance":
            selected = self._min_variance_selection(num_select)
        else:
            # Default to random
            selected = np.random.choice(
                self.num_clients,
                size=num_select,
                replace=False,
            ).tolist()

        # Update counts
        for client_id in selected:
            self.selection_counts[client_id] += 1

        self.record_selection(selected)
        return selected

    def _equal_participation_selection(self, num_select: int) -> List[int]:
        """Select clients to equalize participation over time.

        Prioritizes clients that have been selected less frequently.

        Args:
            num_select: Number of clients to select

        Returns:
            List of selected client IDs
        """
        # Compute inverse selection probability (prioritize underselected)
        inverse_counts = 1.0 / (self.selection_counts + 1)

        # Select based on inverse counts (with some randomness)
        probabilities = inverse_counts / inverse_counts.sum()

        selected = np.random.choice(
            self.num_clients,
            size=num_select,
            replace=False,
            p=probabilities,
        ).tolist()

        return selected

    def _min_variance_selection(self, num_select: int) -> List[int]:
        """Select clients to minimize variance in selection counts.

        Args:
            num_select: Number of clients to select

        Returns:
            List of selected client IDs
        """
        # Sort by selection count and pick least selected
        sorted_indices = np.argsort(self.selection_counts)

        # Add some randomness among least selected
        candidate_pool_size = min(num_select * 3, self.num_clients)
        candidate_pool = sorted_indices[:candidate_pool_size]

        selected = np.random.choice(
            candidate_pool,
            size=num_select,
            replace=False,
        ).tolist()

        return selected


class QualityFairnessSelector(ClientSelector):
    """Balance between model quality and fairness."""

    def __init__(
        self,
        num_clients: int,
        alpha: float = 0.5,
        **kwargs,
    ):
        """Initialize quality-fairness selector.

        Args:
            num_clients: Total number of clients
            alpha: Weight for fairness (0 = only quality, 1 = only fairness)
            **kwargs: Additional configuration
        """
        super().__init__(num_clients, **kwargs)
        self.alpha = alpha
        self.selection_counts = np.zeros(num_clients)

    def select(
        self,
        num_select: int,
        client_info: Optional[Dict[int, Dict[str, Any]]] = None,
        **kwargs,
    ) -> List[int]:
        """Select clients balancing quality and fairness.

        Args:
            num_select: Number of clients to select
            client_info: Client information with quality metrics
            **kwargs: Additional arguments

        Returns:
            List of selected client IDs
        """
        # Compute quality scores (e.g., based on data size or loss)
        if client_info is not None:
            quality_scores = np.array([
                client_info.get(i, {}).get("data_size", 1.0)
                for i in range(self.num_clients)
            ])
        else:
            quality_scores = np.ones(self.num_clients)

        # Normalize quality scores
        quality_scores = quality_scores / quality_scores.sum()

        # Compute fairness scores (inverse of selection frequency)
        fairness_scores = 1.0 / (self.selection_counts + 1)
        fairness_scores = fairness_scores / fairness_scores.sum()

        # Combined score
        combined_scores = (
            (1 - self.alpha) * quality_scores +
            self.alpha * fairness_scores
        )

        # Sample based on combined scores
        probabilities = combined_scores / combined_scores.sum()

        selected = np.random.choice(
            self.num_clients,
            size=num_select,
            replace=False,
            p=probabilities,
        ).tolist()

        # Update counts
        for client_id in selected:
            self.selection_counts[client_id] += 1

        self.record_selection(selected)
        return selected


class GroupFairnessSelector(ClientSelector):
    """Ensure fairness across predefined client groups."""

    def __init__(
        self,
        num_clients: int,
        client_groups: Dict[int, int],
        **kwargs,
    ):
        """Initialize group fairness selector.

        Args:
            num_clients: Total number of clients
            client_groups: Dictionary mapping client ID to group ID
            **kwargs: Additional configuration
        """
        super().__init__(num_clients, **kwargs)
        self.client_groups = client_groups
        self.unique_groups = set(client_groups.values())

    def select(
        self,
        num_select: int,
        client_info: Optional[Dict[int, Dict[str, Any]]] = None,
        **kwargs,
    ) -> List[int]:
        """Select clients ensuring group representation.

        Args:
            num_select: Number of clients to select
            client_info: Optional client information
            **kwargs: Additional arguments

        Returns:
            List of selected client IDs
        """
        selected = []

        # Calculate clients per group
        clients_per_group = num_select // len(self.unique_groups)
        remaining = num_select % len(self.unique_groups)

        for group_id in self.unique_groups:
            # Get clients in this group
            group_clients = [
                cid for cid in range(self.num_clients)
                if self.client_groups.get(cid) == group_id
            ]

            if not group_clients:
                continue

            # Number to select from this group
            n_select = clients_per_group
            if remaining > 0:
                n_select += 1
                remaining -= 1

            # Select randomly from group
            n_select = min(n_select, len(group_clients))
            group_selected = np.random.choice(
                group_clients,
                size=n_select,
                replace=False,
            )
            selected.extend(group_selected.tolist())

        self.record_selection(selected)
        return selected
