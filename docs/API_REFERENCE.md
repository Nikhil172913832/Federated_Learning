# API Reference

Complete API documentation for the Federated Learning Platform.

## Table of Contents

- [Client API](#client-api)
- [Server API](#server-api)
- [Configuration API](#configuration-api)
- [Data Utilities](#data-utilities)
- [Privacy Utilities](#privacy-utilities)
- [Storage API](#storage-api)
- [Tracking API](#tracking-api)

---

## Client API

### ClientApp

Main client application for federated learning.

#### `@app.train()`

Decorator for client training function.

```python
from flwr.app import Message, Context
from flwr.clientapp import ClientApp

app = ClientApp()

@app.train()
def train(msg: Message, context: Context) -> Message:
    """
    Train model on local data.
    
    Args:
        msg (Message): Incoming message containing:
            - content["arrays"]: Global model parameters as ArrayRecord
            - content["config"]: Training configuration (learning rate, etc.)
        context (Context): Execution context containing:
            - run_config: Dictionary with training parameters
            - node_config: Client-specific configuration
                - partition-id: Client's data partition ID
                - num-partitions: Total number of data partitions
    
    Returns:
        Message: Response containing:
            - content["arrays"]: Updated local model parameters
            - content["metrics"]: Training metrics (loss, dataset size)
    
    Example:
        >>> @app.train()
        >>> def train(msg, context):
        ...     model = Net()
        ...     model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        ...     # ... training logic ...
        ...     return Message(content={
        ...         "arrays": ArrayRecord(model.state_dict()),
        ...         "metrics": MetricRecord({"loss": 0.5, "num-examples": 1000})
        ...     })
    """
```

#### `@app.evaluate()`

Decorator for client evaluation function.

```python
@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """
    Evaluate model on local validation data.
    
    Args:
        msg (Message): Message containing model to evaluate
            - content["arrays"]: Model parameters
        context (Context): Execution context
    
    Returns:
        Message: Response with evaluation metrics
            - content["metrics"]: Dict with "eval_loss", "eval_acc", "num-examples"
    
    Example:
        >>> @app.evaluate()
        >>> def evaluate(msg, context):
        ...     model = Net()
        ...     model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
        ...     loss, acc = test_fn(model, valloader, device)
        ...     return Message(content={
        ...         "metrics": MetricRecord({
        ...             "eval_loss": loss,
        ...             "eval_acc": acc,
        ...             "num-examples": len(valloader.dataset)
        ...         })
        ...     })
    """
```

---

## Server API

### ServerApp

Main server application for federated aggregation.

#### `@app.main()`

Decorator for server main function.

```python
from flwr.serverapp import Grid, ServerApp, Context
from flwr.serverapp.strategy import FedAvg

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Main server orchestration for federated learning.
    
    Args:
        grid (Grid): SuperNode grid for client communication
        context (Context): Server execution context
            - run_config: Training configuration
                - "num-server-rounds": Number of FL rounds
                - "fraction-train": Fraction of clients per round
                - "lr": Learning rate
    
    Example:
        >>> @app.main()
        >>> def main(grid, context):
        ...     strategy = FedAvg(fraction_train=0.5)
        ...     global_model = Net()
        ...     
        ...     result = strategy.start(
        ...         grid=grid,
        ...         initial_arrays=ArrayRecord(global_model.state_dict()),
        ...         train_config=ConfigRecord({"lr": 0.01}),
        ...         num_rounds=10
        ...     )
        ...     
        ...     torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
    """
```

### Aggregation Strategies

#### FedAvg

Federated averaging strategy.

```python
from flwr.serverapp.strategy import FedAvg

strategy = FedAvg(
    fraction_train: float = 0.5,      # Fraction of clients to sample per round
    fraction_eval: float = 0.5,       # Fraction for evaluation
    min_train_clients: int = 2,       # Minimum clients for training
    min_eval_clients: int = 2,        # Minimum clients for evaluation
)

# Start federated learning
result = strategy.start(
    grid=grid,                        # SuperNode grid
    initial_arrays=arrays,            # Initial model parameters
    train_config=config,              # Training configuration
    num_rounds=100,                   # Number of FL rounds
)
```

---

## Configuration API

### load_run_config()

Load configuration from YAML or JSON file.

```python
from fl.config import load_run_config

def load_run_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load run configuration from file.
    
    Precedence:
    1. Explicit config_path parameter
    2. Environment variable FL_CONFIG_PATH
    3. Default: complete/fl/config/default.yaml
    
    Args:
        config_path (Optional[str]): Path to configuration file
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If unsupported file format
    
    Example:
        >>> config = load_run_config()
        >>> lr = config["train"]["lr"]  # 0.01
        >>> 
        >>> # Custom config
        >>> config = load_run_config("experiments/config1.yaml")
        >>> # Or via environment
        >>> os.environ["FL_CONFIG_PATH"] = "custom.yaml"
        >>> config = load_run_config()
    """
```

### set_global_seeds()

Set random seeds for reproducibility.

```python
from fl.config import set_global_seeds

def set_global_seeds(seed: int) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch.
    
    Args:
        seed (int): Random seed value
    
    Side Effects:
        - Sets random.seed(seed)
        - Sets np.random.seed(seed)
        - Sets torch.manual_seed(seed)
        - Sets torch.cuda.manual_seed_all(seed)
        - Enables deterministic CUDNN
    
    Example:
        >>> set_global_seeds(42)
        >>> # All random operations now reproducible
        >>> x = torch.randn(10)  # Same values every run
    """
```

### merge_with_context_defaults()

Merge file config with Flower context config.

```python
from fl.config import merge_with_context_defaults

def merge_with_context_defaults(
    context_cfg: Dict[str, Any],
    file_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge configurations with file config taking precedence.
    
    Args:
        context_cfg: Configuration from Flower Context
        file_cfg: Configuration from YAML/JSON file
    
    Returns:
        Dict[str, Any]: Merged configuration
    
    Example:
        >>> context = {"lr": 0.001, "num-server-rounds": 10}
        >>> file = {"lr": 0.01, "local_epochs": 5}
        >>> merged = merge_with_context_defaults(context, file)
        >>> # {'lr': 0.01, 'num-server-rounds': 10, 'local-epochs': 5}
    """
```

---

## Data Utilities

### load_data()

Load partitioned federated dataset.

```python
from fl.task import load_data

def load_data(
    partition_id: int,
    num_partitions: int
) -> Tuple[DataLoader, DataLoader]:
    """
    Load data partition for a specific client.
    
    Args:
        partition_id (int): Client's partition ID (0-indexed)
        num_partitions (int): Total number of partitions
    
    Returns:
        Tuple[DataLoader, DataLoader]: (trainloader, testloader)
            - trainloader: 80% of client's data
            - testloader: 20% of client's data
    
    Example:
        >>> # Client 0 out of 10 total clients
        >>> trainloader, testloader = load_data(0, 10)
        >>> for batch in trainloader:
        ...     images = batch["image"]
        ...     labels = batch["label"]
        ...     # ... training ...
    """
```

### Net (Model)

Simple CNN model for image classification.

```python
from fl.task import Net

class Net(nn.Module):
    """
    Simple CNN for image classification.
    
    Architecture:
        - Conv2D(1, 6, kernel=5) + ReLU + MaxPool
        - Conv2D(6, 16, kernel=5) + ReLU + MaxPool
        - Flatten
        - Linear(256, 120) + ReLU
        - Linear(120, 84) + ReLU
        - Linear(84, 10)
    
    Example:
        >>> model = Net()
        >>> x = torch.randn(32, 1, 28, 28)  # batch_size=32
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 10])
    """
```

### train()

Train model on local data.

```python
from fl.task import train

def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    global_state_dict: Optional[Dict] = None
) -> float:
    """
    Train neural network on local data.
    
    Args:
        net: PyTorch model
        trainloader: Training data loader
        epochs: Number of local epochs
        lr: Learning rate
        device: torch.device ("cuda" or "cpu")
        global_state_dict: Global model parameters (for FedProx)
    
    Returns:
        float: Average training loss
    
    Features:
        - Automatic DP-SGD if enabled in config
        - FedProx proximal term if configured
        - GPU acceleration if available
    
    Example:
        >>> model = Net()
        >>> trainloader, _ = load_data(0, 10)
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> loss = train(model, trainloader, epochs=5, lr=0.01, device=device)
        >>> print(f"Training loss: {loss:.4f}")
    """
```

### test()

Evaluate model on test data.

```python
from fl.task import test

def test(
    net: nn.Module,
    testloader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate model on test data.
    
    Args:
        net: PyTorch model
        testloader: Test data loader
        device: torch.device
    
    Returns:
        Tuple[float, float]: (average_loss, accuracy)
    
    Example:
        >>> model = Net()
        >>> _, testloader = load_data(0, 10)
        >>> device = torch.device("cpu")
        >>> loss, accuracy = test(model, testloader, device)
        >>> print(f"Test accuracy: {accuracy*100:.2f}%")
    """
```

---

## Privacy Utilities

### attach_dp_if_enabled()

Attach differential privacy to training.

```python
from fl.dp import attach_dp_if_enabled

def attach_dp_if_enabled(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    trainloader: DataLoader,
    device: torch.device,
    config: Dict[str, Any]
) -> Tuple[Any, nn.Module, torch.optim.Optimizer, DataLoader]:
    """
    Attach DP-SGD mechanism if enabled in configuration.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer (e.g., Adam, SGD)
        trainloader: Training data loader
        device: torch.device
        config: DP configuration dict with keys:
            - enabled (bool): Whether to enable DP
            - noise_multiplier (float): Noise scale
            - max_grad_norm (float): Gradient clipping threshold
            - target_epsilon (Optional[float]): Target privacy budget
            - target_delta (float): Delta parameter
    
    Returns:
        Tuple containing:
            - PrivacyEngine instance (or None if disabled)
            - Modified model
            - Modified optimizer
            - Modified dataloader
    
    Example:
        >>> config = {
        ...     "enabled": True,
        ...     "noise_multiplier": 1.0,
        ...     "max_grad_norm": 1.0,
        ...     "target_epsilon": 3.0,
        ...     "target_delta": 1e-5
        ... }
        >>> engine, model, opt, loader = attach_dp_if_enabled(
        ...     model, optimizer, trainloader, device, config
        ... )
        >>> # Training now has differential privacy
        >>> if engine:
        ...     epsilon = engine.get_epsilon(delta=1e-5)
        ...     print(f"Current ε: {epsilon:.2f}")
    """
```

### mask_state_dict() / unmask_state_dict()

Apply/remove masking for secure aggregation.

```python
from fl.secure import mask_state_dict, unmask_state_dict

def mask_state_dict(
    state_dict: Dict[str, torch.Tensor],
    enabled: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Add random mask to model parameters for secure aggregation.
    
    Args:
        state_dict: PyTorch model state dictionary
        enabled: Whether masking is enabled
    
    Returns:
        Dict[str, torch.Tensor]: Masked state dictionary
    
    Example:
        >>> model = Net()
        >>> masked = mask_state_dict(model.state_dict(), enabled=True)
        >>> # Gradients now protected with random mask
    """

def unmask_state_dict(
    state_dict: Dict[str, torch.Tensor],
    enabled: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Remove masks during aggregation (simplified implementation).
    
    Args:
        state_dict: Masked state dictionary
        enabled: Whether unmasking is needed
    
    Returns:
        Dict[str, torch.Tensor]: Unmasked state dictionary
    """
```

---

## Storage API

### get_client_store()

Get client-specific storage backend.

```python
from fl.storage import get_client_store

def get_client_store(partition_id: int, config: Dict) -> ClientStore:
    """
    Get storage backend for client checkpoints.
    
    Args:
        partition_id: Client's partition ID
        config: Storage configuration with keys:
            - backend: "folder" or "sqlite"
            - root_dir: Root directory for storage
            - sqlite_dir: Directory for SQLite databases
    
    Returns:
        ClientStore: Storage interface
    
    Example:
        >>> config = {"backend": "folder", "root_dir": "./client_stores"}
        >>> store = get_client_store(partition_id=0, config=config)
        >>> 
        >>> # Save checkpoint
        >>> ckpt_path = store.checkpoint_path(round_num=5)
        >>> torch.save(model.state_dict(), ckpt_path)
        >>> 
        >>> # Load checkpoint
        >>> state = torch.load(ckpt_path)
    """
```

### ClientStore

Abstract storage interface.

```python
class ClientStore:
    """Client checkpoint storage interface."""
    
    def checkpoint_path(self, round_num: int) -> str:
        """
        Get path for checkpoint at specific round.
        
        Args:
            round_num: Training round number
        
        Returns:
            str: Path to checkpoint file
        """
    
    def save_checkpoint(self, round_num: int, state_dict: Dict) -> None:
        """Save model checkpoint."""
    
    def load_checkpoint(self, round_num: int) -> Dict:
        """Load model checkpoint."""
    
    def list_checkpoints(self) -> List[int]:
        """List available checkpoint rounds."""
```

---

## Tracking API

### start_run()

Start MLflow tracking run.

```python
from fl.tracking import start_run

@contextmanager
def start_run(experiment: str, run_name: str):
    """
    Context manager for MLflow tracking run.
    
    Args:
        experiment: MLflow experiment name
        run_name: Name for this specific run
    
    Yields:
        ActiveRun: MLflow active run
    
    Example:
        >>> with start_run(experiment="fl", run_name="client-0"):
        ...     log_params({"lr": 0.01, "epochs": 5})
        ...     log_metrics({"loss": 0.5}, step=0)
        ...     log_model(model, "model")
    """
```

### log_params()

Log hyperparameters to MLflow.

```python
from fl.tracking import log_params

def log_params(params: Dict[str, Any]) -> None:
    """
    Log parameters to MLflow.
    
    Args:
        params: Dictionary of parameter names and values
    
    Example:
        >>> log_params({
        ...     "learning_rate": 0.01,
        ...     "batch_size": 32,
        ...     "optimizer": "adam",
        ...     "local_epochs": 5
        ... })
    """
```

### log_metrics()

Log metrics to MLflow.

```python
from fl.tracking import log_metrics

def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """
    Log metrics to MLflow.
    
    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number (e.g., round number)
    
    Example:
        >>> for round_num in range(10):
        ...     # Training...
        ...     log_metrics({
        ...         "train_loss": 0.5,
        ...         "train_accuracy": 0.85,
        ...         "val_loss": 0.6,
        ...         "val_accuracy": 0.82
        ...     }, step=round_num)
    """
```

### log_model()

Log model to MLflow.

```python
from fl.tracking import log_model

def log_model(model: nn.Module, artifact_path: str) -> None:
    """
    Log PyTorch model to MLflow.
    
    Args:
        model: PyTorch model
        artifact_path: Path in MLflow artifact store
    
    Example:
        >>> model = Net()
        >>> # After training...
        >>> with start_run(experiment="fl", run_name="final"):
        ...     log_model(model, "final_model")
    """
```

---

## Partitioning API

### build_partitioner()

Build data partitioner from configuration.

```python
from fl.partitioning import build_partitioner

def build_partitioner(
    num_partitions: int,
    cfg: Optional[Dict]
) -> Partitioner:
    """
    Build data partitioner based on configuration.
    
    Args:
        num_partitions: Number of client partitions
        cfg: Partitioning configuration:
            - type: "iid", "label_skew", "quantity_skew", "dirichlet"
            - params: Type-specific parameters
    
    Returns:
        Partitioner: Flower dataset partitioner
    
    Example:
        >>> # IID partitioning
        >>> partitioner = build_partitioner(10, None)
        >>> 
        >>> # Label skew
        >>> cfg = {
        ...     "type": "label_skew",
        ...     "params": {"num_labels_per_client": 2}
        ... }
        >>> partitioner = build_partitioner(10, cfg)
        >>> 
        >>> # Dirichlet
        >>> cfg = {
        ...     "type": "dirichlet",
        ...     "params": {"alpha": 0.5}
        ... }
        >>> partitioner = build_partitioner(10, cfg)
    """
```

---

## Personalization API

### fedprox_loss()

Compute FedProx proximal term.

```python
from fl.personalization import fedprox_loss

def fedprox_loss(
    model: nn.Module,
    global_params: List[torch.Tensor],
    mu: float
) -> torch.Tensor:
    """
    Compute FedProx proximal regularization term.
    
    Formula: (mu/2) * ||θ - θ_global||²
    
    Args:
        model: Current local model
        global_params: Global model parameters
        mu: Proximal term coefficient
    
    Returns:
        torch.Tensor: Proximal loss term
    
    Example:
        >>> global_params = [p.clone() for p in model.parameters()]
        >>> # Local training...
        >>> for batch in trainloader:
        ...     optimizer.zero_grad()
        ...     output = model(x)
        ...     loss = criterion(output, y)
        ...     loss += fedprox_loss(model, global_params, mu=0.01)
        ...     loss.backward()
        ...     optimizer.step()
    """
```

---

## Usage Examples

### Complete Training Example

```python
from fl.client_app import app
from fl.task import Net, load_data, train as train_fn, test as test_fn
from fl.config import load_run_config, set_global_seeds
from fl.tracking import start_run, log_params, log_metrics

@app.train()
def train(msg, context):
    # Setup
    config = load_run_config()
    set_global_seeds(config["seed"])
    
    # Load model
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)
    
    # Train
    with start_run(experiment="fl", run_name=f"client-{partition_id}"):
        log_params({"partition_id": partition_id})
        
        loss = train_fn(
            model, trainloader,
            epochs=5, lr=0.01,
            device=device
        )
        
        log_metrics({"train_loss": loss}, step=context.run_config["round"])
    
    # Return update
    return Message(content={
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({"train_loss": loss, "num-examples": len(trainloader.dataset)})
    })
```

---

## Error Handling

All functions may raise the following exceptions:

- `FileNotFoundError`: Configuration or data files not found
- `ValueError`: Invalid configuration or parameters
- `RuntimeError`: Training failures, device errors
- `ConnectionError`: MLflow or SuperLink communication issues

Example error handling:

```python
try:
    config = load_run_config("config.yaml")
except FileNotFoundError:
    print("Config file not found, using defaults")
    config = {}
except ValueError as e:
    print(f"Invalid config: {e}")
    raise
```

---

## Environment Variables

- `FL_CONFIG_PATH`: Path to configuration file
- `MLFLOW_TRACKING_URI`: MLflow server URL (default: http://mlflow:5000)
- `PARTITION_ID`: Client partition ID
- `NUM_PARTITIONS`: Total number of partitions
- `SUPERLINK_ADDRESS`: SuperLink server address

---

## Type Hints

All functions include comprehensive type hints for better IDE support:

```python
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch.utils.data import DataLoader

def train(
    net: torch.nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    global_state_dict: Optional[Dict[str, torch.Tensor]] = None
) -> float:
    ...
```
