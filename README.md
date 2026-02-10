# Federated Learning Platform

Production-ready federated learning platform with privacy-preserving distributed training, communication efficiency, and Byzantine robustness.

## Features

**Algorithms**
- Hybrid gradient compression (20-50x ratio)
- Byzantine-robust aggregation (Multi-Krum, Trimmed Mean)
- Differential privacy (DP-SGD)
- Membership inference attack validation

**Infrastructure**
- Kubernetes deployment with Helm
- CI/CD pipeline with GitHub Actions
- Prometheus + Grafana monitoring
- MLflow experiment tracking

## Quick Start

```bash
./launch-platform.sh
```

Access:
- Dashboard: http://localhost:8050
- MLflow: http://localhost:5000

## Core Structure

```
complete/fl/
├── fl/
│   ├── task.py          # Training loop
│   ├── server_app.py    # Server aggregation
│   ├── client_app.py    # Client training
│   ├── compression.py   # Gradient compression
│   ├── robust_aggregation.py  # Byzantine robustness
│   └── privacy/         # Privacy validation
├── config/
│   └── default.yaml     # Configuration
└── tests/
    └── test_*.py        # Test suite
```

## Configuration

Edit `complete/fl/config/default.yaml`:

```yaml
topology:
  num_clients: 10
  fraction: 0.5

train:
  lr: 0.01
  local_epochs: 1
  num_server_rounds: 10

data:
  dataset: "albertvillanova/medmnist-v2"
  subset: "pneumoniamnist"
  batch_size: 32

privacy:
  dp_sgd:
    enabled: true
    noise_multiplier: 0.8
    target_epsilon: 3.0
```

## Development

Local setup:
```bash
cd complete/fl
pip install -e ".[dev]"
pytest tests/ -v --cov=fl
flwr run . local-simulation --stream
```

Docker:
```bash
./launch-platform.sh
docker compose -f complete/compose-with-ui.yml down
```

## Documentation

- [Architecture](docs/architecture.md) - System design
- [API Reference](docs/api.md) - Module documentation
- [Kubernetes Deployment](docs/kubernetes_deployment.md) - Production deployment
- [Troubleshooting](docs/troubleshooting.md) - Common issues

## Testing

```bash
cd complete/fl
pytest tests/ -v --cov=fl --cov-report=html
```

## License

MIT
