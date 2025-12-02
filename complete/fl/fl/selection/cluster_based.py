"""Cluster-based client selection strategy."""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.cluster import KMeans

from fl.selection.selector import ClientSelector


class ClusterBasedSelector(ClientSelector):
    """Select clients to ensure diversity through clustering.
    
    Groups clients into clusters and selects representatives from each cluster.
    """

    def __init__(
        self,
        num_clients: int,
        num_clusters: Optional[int] = None,
        **kwargs,
    ):
        """Initialize cluster-based selector.

        Args:
            num_clients: Total number of clients
            num_clusters: Number of clusters (default: sqrt(num_clients))
            **kwargs: Additional configuration
        """
        super().__init__(num_clients, **kwargs)
        self.num_clusters = num_clusters or max(int(np.sqrt(num_clients)), 2)
        self.client_clusters: Optional[np.ndarray] = None

    def select(
        self,
        num_select: int,
        client_info: Optional[Dict[int, Dict[str, Any]]] = None,
        **kwargs,
    ) -> List[int]:
        """Select diverse clients using clustering.

        Args:
            num_select: Number of clients to select
            client_info: Client information with features for clustering
            **kwargs: Additional arguments

        Returns:
            List of selected client IDs
        """
        if client_info is None or self.client_clusters is None:
            # Need to cluster clients first
            if client_info is not None:
                self._cluster_clients(client_info)

        if self.client_clusters is None:
            # Fall back to random selection
            return np.random.choice(
                self.num_clients,
                size=num_select,
                replace=False,
            ).tolist()

        # Select from each cluster proportionally
        selected = self._select_from_clusters(num_select)

        self.record_selection(selected)
        return selected

    def _cluster_clients(self, client_info: Dict[int, Dict[str, Any]]) -> None:
        """Cluster clients based on their features.

        Args:
            client_info: Client information dictionary
        """
        # Extract features for clustering
        features = []
        client_ids = []

        for client_id in range(self.num_clients):
            if client_id in client_info:
                info = client_info[client_id]
                # Use data size and loss as features
                feature = [
                    info.get("data_size", 0),
                    info.get("loss", 0),
                ]
                features.append(feature)
                client_ids.append(client_id)

        if not features:
            return

        # Perform clustering
        features_array = np.array(features)
        n_clusters = min(self.num_clusters, len(features))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_array)

        # Map clusters to all clients
        self.client_clusters = np.zeros(self.num_clients, dtype=int)
        for idx, client_id in enumerate(client_ids):
            self.client_clusters[client_id] = clusters[idx]

    def _select_from_clusters(self, num_select: int) -> List[int]:
        """Select clients from different clusters.

        Args:
            num_select: Number of clients to select

        Returns:
            List of selected client IDs
        """
        selected = []

        # Count clients per cluster
        unique_clusters = np.unique(self.client_clusters)
        cluster_counts = {c: np.sum(self.client_clusters == c) for c in unique_clusters}

        # Determine how many to select from each cluster
        clients_per_cluster = num_select // len(unique_clusters)
        remaining = num_select % len(unique_clusters)

        for cluster_id in unique_clusters:
            cluster_clients = np.where(self.client_clusters == cluster_id)[0]

            # Number to select from this cluster
            n_select = clients_per_cluster
            if remaining > 0:
                n_select += 1
                remaining -= 1

            # Select randomly from cluster
            n_select = min(n_select, len(cluster_clients))
            cluster_selected = np.random.choice(
                cluster_clients,
                size=n_select,
                replace=False,
            )
            selected.extend(cluster_selected.tolist())

        return selected[:num_select]


class DiversityMaximizingSelector(ClientSelector):
    """Select clients to maximize diversity in representation."""

    def __init__(self, num_clients: int, **kwargs):
        """Initialize diversity maximizing selector.

        Args:
            num_clients: Total number of clients
            **kwargs: Additional configuration
        """
        super().__init__(num_clients, **kwargs)

    def select(
        self,
        num_select: int,
        client_info: Optional[Dict[int, Dict[str, Any]]] = None,
        **kwargs,
    ) -> List[int]:
        """Select clients to maximize diversity.

        Uses a greedy algorithm to select clients that are most different
        from already selected clients.

        Args:
            num_select: Number of clients to select
            client_info: Client information with features
            **kwargs: Additional arguments

        Returns:
            List of selected client IDs
        """
        if client_info is None:
            # Random fallback
            return np.random.choice(
                self.num_clients,
                size=num_select,
                replace=False,
            ).tolist()

        # Extract feature vectors
        features = self._extract_features(client_info)

        # Greedy selection for maximum diversity
        selected = []
        remaining = list(range(self.num_clients))

        # Select first client randomly
        first_client = np.random.choice(remaining)
        selected.append(first_client)
        remaining.remove(first_client)

        # Greedily select clients that maximize distance to selected set
        while len(selected) < num_select and remaining:
            max_min_distance = -1
            best_client = None

            for candidate in remaining:
                # Compute minimum distance to selected clients
                min_distance = min(
                    np.linalg.norm(features[candidate] - features[s])
                    for s in selected
                )

                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_client = candidate

            if best_client is not None:
                selected.append(best_client)
                remaining.remove(best_client)

        self.record_selection(selected)
        return selected

    def _extract_features(
        self,
        client_info: Dict[int, Dict[str, Any]],
    ) -> Dict[int, np.ndarray]:
        """Extract feature vectors for clients.

        Args:
            client_info: Client information

        Returns:
            Dictionary mapping client ID to feature vector
        """
        features = {}
        for client_id in range(self.num_clients):
            if client_id in client_info:
                info = client_info[client_id]
                feature = np.array([
                    info.get("data_size", 0),
                    info.get("loss", 0),
                    info.get("accuracy", 0),
                ])
            else:
                feature = np.zeros(3)
            features[client_id] = feature

        return features
