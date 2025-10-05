"""Partitioning utilities for IID and non-IID client data splits.

Supports:
- IID
- Label skew via Dirichlet distribution
- Quantity skew via shard-based partitioning
- Covariate shift via per-client transform variants (handled in task.py)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from flwr_datasets.partitioner import (
    IidPartitioner,
    DirichletPartitioner,
    ShardPartitioner,
)


@dataclass
class PartitionerSpec:
    name: str
    kwargs: Dict[str, Any]


def build_partitioner(num_partitions: int, cfg: Optional[Dict[str, Any]]) -> Any:
    """Return a Flower partitioner based on config.

    cfg example:
    {"type": "label_skew", "params": {"alpha": 0.3}}
    {"type": "quantity_skew", "params": {"min_size": 100}}
    {"type": "iid"}
    """

    if not cfg or cfg.get("type", "iid") == "iid":
        return IidPartitioner(num_partitions=num_partitions)

    ptype = cfg.get("type")
    params = cfg.get("params", {})

    if ptype == "label_skew":
        alpha = float(params.get("alpha", 0.5))
        return DirichletPartitioner(num_partitions=num_partitions, alpha=alpha)

    if ptype == "quantity_skew":
        # Use shard-based partitioning for quantity skew
        # Each client gets a different number of shards, creating heterogeneity
        shard_size = int(params.get("shard_size", 100))
        return ShardPartitioner(
            num_partitions=num_partitions,
            partition_by="train",
            shard_size=shard_size,
            shuffle=True,
        )

    # Fallback to IID if unknown type
    return IidPartitioner(num_partitions=num_partitions)


