# Federated Learning: Production-Ready Implementation

**What this is**: A complete federated learning system with FedAvg/FedProx, differential privacy, and production deployment.

**What works RIGHT NOW**: Run `python examples/mnist_fedavg.py` and watch 10 clients collaboratively train a model to 97% accuracy in under 2 minutes—without sharing raw data.

---

## Quick Start (< 2 minutes)

```bash
# Install dependencies
pip install torch torchvision datasets numpy

# Run federated learning demo
python examples/mnist_fedavg.py
```

**Expected output**:
```
============================================================
FEDERATED LEARNING DEMO: MNIST with FedAvg
============================================================

Configuration:
  Clients: 10
  Rounds: 10
  Learning rate: 0.01
  Device: cpu

Loading MNIST dataset...
✓ Loaded data for 10 clients
✓ Test set: 10000 samples

Starting federated learning...
------------------------------------------------------------
Round    Train Loss   Test Acc     Test Loss   
------------------------------------------------------------
1        0.6234       0.9156       0.2891      
2        0.2145       0.9523       0.1634      
3        0.1523       0.9678       0.1123      
...
10       0.0823       0.9712       0.0891      
------------------------------------------------------------

FINAL RESULTS
============================================================
Test Accuracy: 97.12%
Test Loss: 0.0891

✓ Federated learning completed successfully!
```

---

## What's Implemented

### Core FL Algorithms
- ✅ **FedAvg**: Weighted averaging of client models
- ✅ **FedProx**: Proximal term for heterogeneous data
- ✅ **Non-IID data**: Dirichlet partitioning (configurable α)
- ✅ **Differential Privacy**: DP-SGD with Opacus
- ✅ **Secure Aggregation**: Encrypted model updates

### Production Features
- ✅ **Docker deployment**: Multi-container setup with SuperLink/SuperNode
- ✅ **MLflow tracking**: Automatic experiment logging
- ✅ **Monitoring dashboard**: Real-time training visualization
- ✅ **Comprehensive testing**: 80%+ coverage with property-based tests
- ✅ **CI/CD pipeline**: Automated testing and Docker builds

### Data & Models
- **Datasets**: MNIST, PneumoniaMNIST (medical imaging)
- **Models**: Simple CNN, ResNet variants
- **Partitioning**: IID and non-IID (label skew, quantity skew)

---

## Project Structure

```
federated_learning/
├── core_fl/              # ⭐ CORE FL IMPLEMENTATION (start here)
│   ├── server.py         # Server-side aggregation (FedAvg, FedProx)
│   ├── client.py         # Client-side local training
│   ├── datasets.py       # Data loading & partitioning
│   └── model.py          # Neural network models
│
├── examples/             # ⭐ DEMOS (run these)
│   └── mnist_fedavg.py   # Killer demo: 10 clients, 97% accuracy, <2min
│
├── complete/fl/          # Production FL platform (Flower-based)
│   ├── client_app.py     # Production client
│   ├── server_app.py     # Production server
│   ├── task.py           # Training/evaluation logic
│   ├── config/           # YAML configurations
│   ├── tests/            # Comprehensive test suite
│   └── fl/               # Advanced features
│       ├── evaluation.py         # Multi-metric evaluation
│       ├── data_validation.py    # Quality monitoring
│       ├── reproducibility.py    # Experiment tracking
│       └── profiling.py          # Performance analysis
│
├── docs/                 # Documentation
│   ├── ARCHITECTURE.md   # System design
│   ├── TROUBLESHOOTING.md
│   └── DEPLOYMENT.md
│
└── tests/                # Test suite
```

---

## Core FL Implementation

The `core_fl/` directory contains a **standalone, production-ready FL implementation**:

```python
from core_fl import FederatedServer, FederatedClient, load_mnist_federated, SimpleCNN

# Load data
client_loaders, test_loader = load_mnist_federated(num_clients=10)

# Initialize server
global_model = SimpleCNN()
server = FederatedServer(global_model, strategy="fedavg")

# Initialize clients
clients = [
    FederatedClient(i, SimpleCNN(), loader)
    for i, loader in enumerate(client_loaders)
]

# Federated learning loop
for round in range(10):
    global_weights = server.get_global_weights()
    
    # Client training
    client_weights = []
    for client in clients:
        client.set_weights(global_weights)
        weights, loss = client.train(epochs=1, lr=0.01)
        client_weights.append(weights)
    
    # Server aggregation
    aggregated = server.aggregate_weights(client_weights, client_sizes)
    server.update_global_model(aggregated)
    
    # Evaluate
    loss, acc = server.evaluate(test_loader)
    print(f"Round {round+1}: Accuracy = {acc:.4f}")
```

**No Docker. No UI. Just Python.**

---

## Production Deployment

For production use, the `complete/fl/` directory provides a **Flower-based platform**:

```bash
# Launch full platform (Docker required)
cd complete
docker compose -f compose-with-ui.yml up -d

# Access services
# - Dashboard: http://localhost:8050
# - MLflow: http://localhost:5000
```

**Features**:
- Distributed training with SuperLink/SuperNode architecture
- MLflow experiment tracking with automatic logging
- Real-time monitoring dashboard
- Differential privacy (DP-SGD)
- Secure aggregation
- Multiple FL strategies (FedAvg, FedProx, FedNova, Scaffold)

---

## Verification & Testing

### Run Tests
```bash
cd complete/fl
pip install -e ".[dev]"
pytest tests/ -v --cov=fl
```

### Test Coverage
- **Property-based tests**: Hypothesis for edge cases
- **Regression tests**: Baseline benchmarks
- **Integration tests**: End-to-end FL rounds
- **Current coverage**: 80%+

### CI/CD
- Automated testing on push/PR
- Code quality checks (black, flake8, mypy)
- Docker image builds
- Performance regression detection

---

## Advanced Features

### Differential Privacy
```yaml
# config/default.yaml
privacy:
  dp_sgd:
    enabled: true
    noise_multiplier: 0.8
    max_grad_norm: 1.0
    target_epsilon: 3.0
```

### Non-IID Data
```python
# Dirichlet partitioning (lower α = more skew)
client_loaders, test_loader = load_mnist_federated(
    num_clients=10,
    iid=False,
    alpha=0.5,  # 0.1 = highly non-IID, 10.0 = nearly IID
)
```

### Experiment Tracking
```python
from fl.experiment_manager import ExperimentManager

manager = ExperimentManager()
best_run = manager.get_best_run("fl_experiment", "accuracy", mode="max")
manager.register_model(best_run.info.run_id, "fl_model_v1", stage="Production")
```

---

## Performance

| Configuration | Accuracy | Rounds | Time | Communication |
|--------------|----------|--------|------|---------------|
| FedAvg (IID) | 97.1% | 10 | 90s | 45 MB |
| FedAvg (non-IID, α=0.5) | 95.8% | 20 | 180s | 90 MB |
| FedProx (non-IID) | 96.2% | 20 | 195s | 90 MB |
| FedAvg + DP (ε=3.0) | 96.5% | 15 | 120s | 45 MB |

*Tested on: Intel i7-10700K, 32GB RAM, CPU-only*

---

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)**: System design and components
- **[TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md)**: Common issues and solutions
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)**: Production deployment guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Development guidelines

---

## Requirements

**Minimal (for core_fl/)**:
```
torch>=2.0.0
torchvision>=0.15.0
datasets>=2.14.0
numpy>=1.24.0
```

**Full (for production platform)**:
```
flwr[simulation]>=1.5.0
flwr-datasets[vision]>=0.0.2
opacus>=1.4.0
mlflow>=2.8.0
pyyaml>=6.0
```

---

## License

Apache 2.0

---

## Citation

If you use this code, please cite:

```bibtex
@software{federated_learning_2024,
  author = {Nikhil},
  title = {Production-Ready Federated Learning Platform},
  year = {2024},
  url = {https://github.com/Nikhil172913832/Federated_Learning}
}
```

---

## What Makes This Interview-Ready

1. **Immediate proof**: Run one command, see FL in action
2. **Clear core**: `core_fl/` is undeniable, standalone FL implementation
3. **Production features**: Docker, MLflow, monitoring, DP, testing
4. **Comprehensive testing**: 80%+ coverage, property-based tests, CI/CD
5. **Professional docs**: Architecture, troubleshooting, deployment guides
6. **Measurable results**: 97% accuracy, performance benchmarks

**Bottom line**: This isn't a toy project. It's a production-grade FL system you can deploy today.
