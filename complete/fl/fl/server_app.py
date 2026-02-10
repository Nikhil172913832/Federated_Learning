"""fl: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl.models import SimpleCNN
from fl.config import load_run_config, merge_with_context_defaults, set_global_seeds
from fl.secure import unmask_state_dict
from fl.tracking import start_run, log_params, log_metrics
from fl.reproducibility import ensure_reproducibility
from pathlib import Path

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Merge file config with Flower context config then set seeds
    file_cfg = load_run_config()
    run_cfg = merge_with_context_defaults(context.run_config, file_cfg.get("train", {}))
    # Topology overrides
    topo_cfg = file_cfg.get("topology", {})
    if "fraction" in topo_cfg:
        run_cfg["fraction-train"] = topo_cfg["fraction"]
    if "num_server_rounds" in file_cfg.get("train", {}):
        run_cfg["num-server-rounds"] = file_cfg["train"]["num_server_rounds"]
    if "seed" in file_cfg:
        set_global_seeds(int(file_cfg["seed"]))

    fraction_train: float = float(run_cfg["fraction-train"])
    num_rounds: int = int(run_cfg["num-server-rounds"])
    lr: float = float(run_cfg["lr"])

    # Load global model
    global_model = SimpleCNN()
    arrays = ArrayRecord(
        unmask_state_dict(
            global_model.state_dict(),
            enabled=bool(file_cfg.get("security", {}).get("secure_agg", False)),
        )
    )

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train)

    # Start MLflow run for server
    with start_run(experiment="fl", run_name="server"):
        # Ensure reproducibility - create and save manifest
        manifest_dir = Path("experiments") / "server_manifests"
        manifest = ensure_reproducibility(file_cfg, manifest_dir)

        log_params(
            {
                "num_rounds": num_rounds,
                "fraction_train": fraction_train,
                "lr": lr,
                "num_partitions": 3,  # from the supernode configs
                "seed": file_cfg.get("seed", 42),
                "git_commit": manifest.git_commit or "N/A",
                "dependencies_hash": manifest.dependencies_hash[:16],
            }
        )

        # Start strategy, run FedAvg for `num_rounds`
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({"lr": lr}),
            num_rounds=num_rounds,
        )

        # Log final metrics
        log_metrics({"training_complete": 1}, step=num_rounds)

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
    print(f"Model saved! Training completed with {num_rounds} rounds.")
