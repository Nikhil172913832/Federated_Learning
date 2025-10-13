# Federated Learning Platform

A production-ready federated learning platform using Flower framework with Docker containerized clients, real-time monitoring dashboard, and MLflow experiment tracking.

## Features

- Containerized FL clients (3 Docker containers)
- Containerized FL server
- Real-time monitoring dashboard
- MLflow experiment tracking
- Config-driven experiments
- Non-IID data partitioning
- Differential privacy (DP-SGD)
- FedProx personalization

## Prerequisites

- Docker and Docker Compose V2
- Python 3.10+
- `flwr` CLI: `pip install flwr`

## Quick Start

1. Launch the platform:
```bash
./launch-platform.sh
```

2. Verify all services are running:
```bash
./verify-platform.sh
```

3. Start training:
```bash
cd complete
flwr run fl local-deployment --stream
```

## Access Points

- Dashboard: http://localhost:8050
- MLflow: http://localhost:5000
- SuperLink API: http://localhost:9093

## Configuration

Edit `complete/fl/config/default.yaml` to change training parameters:

```yaml
topology:
  num_clients: 10
  fraction: 0.5

train:
  lr: 0.01
  local_epochs: 1
  num_server_rounds: 3

data:
  dataset: albertvillanova/medmnist-v2
  subset: pneumoniamnist
  batch_size: 32
  iid: true
```

You can also update configuration through the dashboard UI.
- **Platform UI**: Real-time monitoring dashboard
- **MLflow**: Experiment tracking and model versioning

### Advanced Features

**State Persistence:**
```bash
docker compose -f compose.yml -f with-state.yml up --build -d
```

**TLS Encryption:**
```bash
# Generate certificates
docker compose -f certs.yml run --rm --build gen-certs

# Start with TLS
docker compose -f compose.yml -f with-tls.yml up --build -d

# Update pyproject.toml for TLS
# Add: [tool.flwr.federations.local-deployment-tls]
#      address = "127.0.0.1:9093"
#      root-certificates = "../superlink-certificates/ca.crt"

# Run with TLS
flwr run quickstart-compose local-deployment-tls --stream
```

**Combined State + TLS:**
```bash
docker compose -f compose.yml -f with-tls.yml -f with-state.yml up --build -d
```


## Architecture

```
SuperLink (Coordination)
    |
    |-- SuperNode-1 --> SuperExec-ClientApp-1 (Docker)
    |-- SuperNode-2 --> SuperExec-ClientApp-2 (Docker)  
    |-- SuperNode-3 --> SuperExec-ClientApp-3 (Docker)
    |-- SuperExec-ServerApp (Docker)
    |
    |-- MLflow (Tracking)
    |-- Dashboard (Monitoring)
```

## Project Structure

```
complete/
├── compose-with-ui.yml          # Main deployment file
├── fl/                          # FL application
│   ├── config/default.yaml      # Training configuration
│   ├── fl/
│   │   ├── client_app.py        # Client logic
│   │   ├── server_app.py        # Server logic
│   │   └── task.py              # Model and data loading
│   └── pyproject.toml
├── Dockerfile                   # FL training containers
└── mlflow.Dockerfile            # MLflow container

platform-ui/
├── app.py                       # Dashboard application
├── Dockerfile
└── requirements.txt
```

## Available Commands

```bash
# Start platform
./launch-platform.sh

# Verify services
./verify-platform.sh

# Quick start helper
./quick-start.sh

# Stop platform
cd complete && docker compose -f compose-with-ui.yml down

# View logs
cd complete && docker compose -f compose-with-ui.yml logs -f

# Restart a service
cd complete && docker compose -f compose-with-ui.yml restart <service-name>
```

## Monitoring

### Dashboard Features
- Real-time container status
- System resource monitoring
- Training log viewer
- Configuration editor
- Start/stop training controls

### MLflow Features
- Experiment tracking
- Metric visualization
- Model comparison
- Run history

## Advanced Configuration

### Non-IID Data
```yaml
data:
  iid: false
  non_iid:
    type: label_skew
    params:
      num_labels_per_client: 2
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

## Troubleshooting

**Services not starting?**
```bash
docker compose -f complete/compose-with-ui.yml down -v
./launch-platform.sh
```

**No logs in MLflow?**
- Wait for first training round to complete
- Check MLFLOW_TRACKING_URI is set in containers
- Verify network connectivity between containers

**Training fails?**
```bash
# Check container logs
docker logs complete-superexec-serverapp-1
docker logs complete-superexec-clientapp-1-1
```

## License

See LICENSE file for details.
