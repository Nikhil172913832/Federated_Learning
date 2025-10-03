"""fl: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl.task import Net
from fl.config import load_run_config, merge_with_context_defaults, set_global_seeds
from fl.secure import unmask_state_dict

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
    global_model = Net()
    arrays = ArrayRecord(unmask_state_dict(global_model.state_dict(), enabled=bool(file_cfg.get("security", {}).get("secure_agg", False))))

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_train=fraction_train)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
