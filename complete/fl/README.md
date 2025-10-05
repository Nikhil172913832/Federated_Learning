# fl: A Federated Learning Flower App

## Summary
A federated learning project using Flower with PyTorch, featuring configurable experiments (YAML/JSON), non-IID client partitions, personalization (FedProx), DP-SGD (Opacus), per-client storage simulation (folder/SQLite), tracking (MLflow), checkpoints, tests, CI, and Docker Compose for deployment.

## Architecture
- `fl/server_app.py`: Server with FedAvg strategy
- `fl/client_app.py`: Client training/eval; config-driven preprocessing, personalization, DP-SGD, tracking, checkpointing
- `fl/task.py`: Model, transforms, loaders, train/eval
- `fl/partitioning.py`: IID and non-IID partitioners
- `fl/personalization.py`: FedProx loss
- `fl/dp.py`: DP-SGD via Opacus
- `fl/storage.py`: Per-client folder/SQLite store
- `fl/tracking.py`: MLflow utilities
- `config/default.yaml`: Config

## Quickstart

### Option 1: Docker Compose (Recommended)
```bash
# Prerequisites: Install flwr CLI and ensure Docker is running
pip install flwr

# Setup Docker Compose environment
git clone --depth=1 --branch v1.22.0 https://github.com/adap/flower.git _tmp \
  && mv _tmp/framework/docker/complete . \
  && rm -rf _tmp

# Create Flower project
flwr new quickstart-compose --framework PyTorch --username flower

# Set environment variable
export PROJECT_DIR=quickstart-compose

# Start services
docker compose up --build -d

# Run the project
flwr run quickstart-compose local-deployment --stream

# Stop services when done
docker compose down
```

### Option 2: Local Simulation
```bash
# Install
pip install -e complete/fl

# Run local simulation
cd complete/fl
flwr run .

# Tests
cd complete/fl
pytest -q

# Optional: MLflow UI
mlflow ui --backend-store-uri file:./mlruns
```

## Non-IID Experiments
Configure `data.non_iid`.
- Label skew: `{type: label_skew, params: {alpha: 0.3}}`
- Quantity skew: `{type: quantity_skew, params: {min_size: 100}}`

## Personalization
Set `personalization.method: fedprox` and `personalization.fedprox_mu`.

## Differential Privacy
Set `privacy.dp_sgd.enabled: true`, configure `noise_multiplier`, `max_grad_norm`. Document epsilon/utility tradeoffs.

## Ethical Note
Use only public datasets (e.g., MedMNIST). No real patient data.

## Reproduce from Checkpoint
```bash
python -m fl.scripts.resume_from_ckpt --ckpt path/to/round_X.pt
```

## Design notes & limitations
- Simulated hospitals with local stores
- Secure aggregation stub provided
- Flower deployment via Docker Compose included
