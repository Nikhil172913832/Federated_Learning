#!/usr/bin/env python3
"""Profile FL training performance.

This script profiles a training run and generates performance reports.
"""

import argparse
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fl.profiling import PerformanceProfiler
from fl.config import load_run_config, set_global_seeds
from fl.models import SimpleCNN
from fl.task import load_data, train


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Profile FL training performance")
    parser.add_argument(
        "--config", type=Path, default="config/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="profiling",
        help="Output directory for profiling results",
    )
    parser.add_argument(
        "--partition-id", type=int, default=0, help="Partition ID to profile"
    )
    parser.add_argument(
        "--num-partitions", type=int, default=3, help="Total number of partitions"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to profile"
    )

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_run_config(str(args.config))

    # Set seeds
    seed = config.get("seed", 42)
    set_global_seeds(seed)

    # Create profiler
    profiler = PerformanceProfiler(args.output)

    # Load model and data
    logger.info("Loading model and data...")
    model = SimpleCNN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trainloader, _ = load_data(args.partition_id, args.num_partitions)

    lr = config.get("train", {}).get("lr", 0.01)

    # Profile training
    logger.info("Starting profiling...")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Learning rate: {lr}")

    profiler.log_memory_stats()

    with profiler.profile_training(
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        with profiler.measure_time("Training"):
            train_loss = train(
                model,
                trainloader,
                epochs=args.epochs,
                lr=lr,
                device=device,
            )

    logger.info(f"Training loss: {train_loss:.4f}")
    profiler.log_memory_stats()

    logger.info(f"\nProfiling results saved to {args.output}")
    logger.info("View trace in Chrome: chrome://tracing")

    return 0


if __name__ == "__main__":
    import torch

    sys.exit(main())
