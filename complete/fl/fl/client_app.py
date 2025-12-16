"""fl: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl.task import Net, load_data
from fl.task import test as test_fn
from fl.task import train as train_fn
from fl.config import load_run_config, merge_with_context_defaults, set_global_seeds
from fl.storage import get_client_store
from fl.secure import mask_state_dict
from fl.tracking import start_run, log_params, log_metrics
from fl.reproducibility import ensure_reproducibility
from pathlib import Path

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Load config and set seeds for reproducibility
    file_cfg = load_run_config()
    if "seed" in file_cfg:
        set_global_seeds(int(file_cfg["seed"]))
    run_cfg = merge_with_context_defaults(context.run_config, file_cfg.get("train", {}))

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    store = get_client_store(partition_id, file_cfg)
    trainloader, _ = load_data(partition_id, num_partitions)

    # Ensure reproducibility - create and save manifest
    manifest_dir = store.root / "manifests"
    manifest = ensure_reproducibility(file_cfg, manifest_dir)

    # Call the training function
    with start_run(experiment="fl", run_name=f"client-{partition_id}"):
        log_params({
            "partition_id": partition_id,
            "num_partitions": num_partitions,
            "local_epochs": int(run_cfg["local-epochs"]),
            "lr": msg.content["config"]["lr"],
            "seed": file_cfg.get("seed", 42),
            "git_commit": manifest.git_commit or "N/A",
            "data_fingerprint": manifest.data_fingerprint[:16] if manifest.data_fingerprint != "N/A" else "N/A",
        })
        train_loss = train_fn(
        model,
        trainloader,
        int(run_cfg["local-epochs"]),
        msg.content["config"]["lr"],
        device,
        global_state_dict=msg.content["arrays"].to_torch_state_dict(),
        )
        round_idx = int(context.run_config.get("round", 0))
        log_metrics({"train_loss": train_loss}, step=round_idx)
        # Save checkpoint
        ckpt_path = store.checkpoint_path(round_idx)
        torch.save(model.state_dict(), ckpt_path)

    # Construct and return reply Message
    secure_enabled = bool(file_cfg.get("security", {}).get("secure_agg", False))
    masked_sd = mask_state_dict(model.state_dict(), enabled=secure_enabled)
    model_record = ArrayRecord(masked_sd)
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load config and set seeds
    file_cfg = load_run_config()
    if "seed" in file_cfg:
        set_global_seeds(int(file_cfg["seed"]))

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    store = get_client_store(partition_id, file_cfg)
    _, valloader = load_data(partition_id, num_partitions)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
