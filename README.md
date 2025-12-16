# Federated Learning: Production-Ready Platform

**What this is**: A complete, production-grade federated learning platform with FedAvg/FedProx, differential privacy, secure aggregation, and real-time monitoring.

**Quick Demo**: Run `./launch-platform.sh` and visit http://localhost:8050 to see FL in action.

---

## Quick Start (2 minutes)

```bash
# Launch the platform
./launch-platform.sh

# Access the dashboard
open http://localhost:8050

# View experiment tracking
open http://localhost:5000
```

**What you'll see**:
- Real-time training progress across multiple clients
- Accuracy improving from ~60% → 95%+ over rounds
- MLflow tracking all experiments automatically
- No raw data shared between clients

---

## Core Implementation

The heart of this FL system is in **`complete/fl/`**:

```
complete/fl/
├── fl/
│   ├── task.py          # ⭐ Core: Model, train(), test(), load_data()
│   ├── client_app.py    # ⭐ Client-side FL logic
│   ├── server_app.py    # ⭐ Server-side aggregation
│   ├── aggregation.py   # FedAvg, FedProx, FedNova, Scaffold
│   ├── personalization.py # FedProx, FedBN, fine-tuning
│   ├── dp.py            # Differential privacy (DP-SGD)
│   ├── secure_agg.py    # Secure aggregation
│   └── evaluation.py    # Comprehensive metrics
├── config/
│   └── default.yaml     # Configuration (dataset, FL params, privacy)
└── tests/
    └── test_*.py        # Comprehensive test suite
```

**Key files to review**:
- [`fl/task.py`](complete/fl/fl/task.py) - Training loop, model, data loading
- [`fl/server_app.py`](complete/fl/fl/server_app.py) - Server aggregation logic
- [`fl/client_app.py`](complete/fl/fl/client_app.py) - Client training logic

---

## What's Implemented

### FL Algorithms
- ✅ **FedAvg**: Weighted averaging (McMahan et al., 2017)
- ✅ **FedProx**: Proximal term for heterogeneous data
- ✅ **FedNova**: Normalized averaging
- ✅ **Scaffold**: Variance reduction

### Privacy & Security
- ✅ **Differential Privacy**: DP-SGD with Opacus (ε-δ guarantees)
- ✅ **Secure Aggregation**: Encrypted model updates
- ✅ **Non-IID data**: Dirichlet partitioning, label skew

### Production Features
- ✅ **Docker deployment**: Multi-container orchestration
- ✅ **MLflow tracking**: Automatic experiment logging
- ✅ **Real-time dashboard**: Live training visualization
- ✅ **Comprehensive testing**: 80%+ coverage
- ✅ **CI/CD pipeline**: Automated quality checks

### Data & Models
- **Datasets**: MNIST, CIFAR-10, PneumoniaMNIST (medical imaging)
- **Models**: CNN, ResNet variants
- **Partitioning**: IID, label skew, quantity skew, covariate shift

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     SuperLink (Server)                   │
│  - Aggregates client updates (FedAvg/FedProx/etc)       │
│  - Manages training rounds                               │
│  - Logs to MLflow                                        │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼──────┐  ┌────────▼─────┐  ┌────────▼─────┐
│  SuperNode 1 │  │ SuperNode 2  │  │ SuperNode N  │
│  (Client)    │  │  (Client)    │  │  (Client)    │
│              │  │              │  │              │
│ Private Data │  │ Private Data │  │ Private Data │
└──────────────┘  └──────────────┘  └──────────────┘
```

**Key Components**:
1. **SuperLink**: Coordinates training, aggregates updates
2. **SuperNodes**: Train on private data, send updates
3. **MLflow**: Tracks experiments, metrics, models
4. **Dashboard**: Real-time visualization

---

## Performance Benchmarks

**Hardware**: Intel i7-10700K, 32GB RAM, NVIDIA RTX 3080

| Configuration | Dataset | Accuracy | Rounds | Time | Communication |
|--------------|---------|----------|--------|------|---------------|
| **FedAvg (IID)** | MNIST | **98.5%** | 10 | 90s | 45 MB |
| FedAvg (non-IID, α=0.5) | MNIST | 96.8% | 20 | 180s | 90 MB |
| FedProx (non-IID) | MNIST | 97.3% | 20 | 195s | 90 MB |
| FedAvg + DP (ε=3.0) | MNIST | 97.2% | 15 | 120s | 45 MB |
| FedAvg (IID) | PneumoniaMNIST | 94.7% | 50 | 300s | 120 MB |
| FedAvg (IID) | CIFAR-10 | 87.3% | 100 | 600s | 200 MB |

**Key Metrics**:
- **Convergence**: 10-20 rounds for MNIST
- **Scalability**: Linear with number of clients
- **Privacy**: DP with ε=3.0 (strong privacy guarantee)
- **Communication**: ~4.5 MB per round (MNIST)

---

## Configuration

Edit `complete/fl/config/default.yaml`:

```yaml
# Federated topology
topology:
  num_clients: 10
  fraction: 0.5  # Sample 50% of clients per round

# Training parameters
train:
  lr: 0.01
  local_epochs: 1
  num_server_rounds: 10

# Dataset
data:
  dataset: "albertvillanova/medmnist-v2"
  subset: "pneumoniamnist"
  batch_size: 32
  
# Non-IID data (optional)
  non_iid:
    type: "label_skew"  # or "quantity_skew", "covariate_shift"
    params:
      alpha: 0.5  # Lower = more skew

# Differential Privacy (optional)
privacy:
  dp_sgd:
    enabled: true
    noise_multiplier: 0.8
    max_grad_norm: 1.0
    target_epsilon: 3.0

# Personalization (optional)
personalization:
  method: "fedprox"  # or "fedbn", "finetune"
  fedprox_mu: 0.01
```

---

## Development

### Local Development (without Docker)

```bash
cd complete/fl

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=fl

# Run linting
black fl/ tests/
flake8 fl/ tests/
mypy fl/

# Run FL simulation locally
flwr run . local-simulation --stream
```

### Docker Development

```bash
cd complete

# Build images
docker compose build

# Run platform
docker compose -f compose-with-ui.yml up -d

# View logs
docker compose logs -f

# Stop platform
docker compose down
```

---

## Testing

```bash
cd complete/fl

# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=fl --cov-report=html

# Specific test
pytest tests/test_integration.py -v

# Property-based tests
pytest tests/test_data_validation.py -v --hypothesis-show-statistics
```

**Test Coverage**: 80%+

---

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: System design and components
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**: Common issues and solutions
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)**: Production deployment guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Development guidelines

---

## Example: Custom FL Algorithm

```python
# complete/fl/fl/custom_strategy.py
from flwr.server.strategy import Strategy

class MyCustomStrategy(Strategy):
    def aggregate_fit(self, server_round, results, failures):
        # Your custom aggregation logic
        weights = [r.parameters for r, _ in results]
        # ... custom aggregation ...
        return aggregated_weights
```

Then update `complete/fl/fl/server_app.py` to use your strategy.

---

## Deployment

### Docker Compose (Recommended)

```bash
./launch-platform.sh
```

### Kubernetes

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for Kubernetes manifests and Helm charts.

### Cloud Deployment

- **AWS**: ECS/EKS deployment guides
- **GCP**: GKE deployment guides  
- **Azure**: AKS deployment guides

---

## What Makes This Production-Ready

1. **Battle-tested algorithms**: FedAvg, FedProx, DP-SGD
2. **Comprehensive testing**: 80%+ coverage, property-based tests
3. **Production deployment**: Docker, Kubernetes, monitoring
4. **Privacy guarantees**: Differential privacy with ε-δ
5. **Real-world datasets**: Medical imaging (PneumoniaMNIST)
6. **Experiment tracking**: MLflow integration
7. **Professional docs**: Architecture, troubleshooting, deployment

---

## Citation

```bibtex
@software{federated_learning_2024,
  author = {Nikhil},
  title = {Production-Ready Federated Learning Platform},
  year = {2024},
  url = {https://github.com/Nikhil172913832/Federated_Learning}
}
```

---

## License

Apache 2.0

---

## Quick Links

- 🚀 [Quick Start](#quick-start-2-minutes)
- 📖 [Core Implementation](#core-implementation)
- 🏗️ [Architecture](#architecture)
- 📊 [Performance](#performance-benchmarks)
- ⚙️ [Configuration](#configuration)
- 🧪 [Testing](#testing)
- 📚 [Documentation](#documentation)
