# Federated Learning Application

Flower-based federated learning application with PyTorch, featuring configurable experiments, non-IID data partitioning, personalization algorithms, differential privacy, and MLflow tracking.

## Features

- Configurable via YAML
- Non-IID data partitioning (label skew, quantity skew)
- Personalization (FedProx)
- Differential privacy (DP-SGD via Opacus)
- Per-client storage simulation
- MLflow experiment tracking
- Checkpoint support

## Project Structure

```
fl/
├── client_app.py      # Client training logic
├── server_app.py      # Server aggregation
├── task.py           # Model and data loading
├── partitioning.py   # Data distribution
├── personalization.py # FedProx implementation
├── dp.py             # Differential privacy
├── storage.py        # Client state persistence
└── tracking.py       # MLflow integration

config/
└── default.yaml      # Training configuration
```

## Local Development

```bash
# Install
pip install -e .

# Run simulation
flwr run .

# Run tests
pytest -q
```

## Configuration Examples

### Non-IID Data
```yaml
data:
  iid: false
  non_iid:
    type: label_skew
    params:
      alpha: 0.3
```

### Differential Privacy
```yaml
privacy:
  dp_sgd:
    enabled: true
    noise_multiplier: 0.8
    max_grad_norm: 1.0
```

### Personalization
```yaml
personalization:
  method: fedprox
  fedprox_mu: 0.01
```

## Checkpoint Resume

```bash
python -m fl.scripts.resume_from_ckpt --ckpt path/to/round_X.pt
```
