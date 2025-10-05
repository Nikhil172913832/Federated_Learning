# 🚀 Federated Learning Platform

A **complete federated learning platform** showcasing **Docker container clients** with real-time monitoring UI. This platform demonstrates production-ready federated learning using the Flower framework.

## 🌟 Platform Features

- **🐳 Docker Container Clients**: Each client runs in a separate Docker container
- **📊 Real-time Monitoring UI**: Web dashboard to monitor the federated learning process
- **🖥️ Containerized Server**: Server runs in its own Docker container
- **🌐 SuperLink Coordination**: Central coordination service for federated learning
- **📈 MLflow Tracking**: Experiment tracking and model versioning
- **⚙️ Config-driven Experiments**: YAML-based configuration system
- **🔒 Security Features**: TLS encryption and state persistence options
- **📋 Non-IID Data**: Realistic client data heterogeneity
- **🎯 Personalization**: FedProx algorithm support
- **🔐 Differential Privacy**: DP-SGD via Opacus

## 🚀 Quick Start - Complete Platform

### Prerequisites
- `flwr` CLI installed locally (`pip install flwr`)
- Docker and Docker Compose V2 installed and running

### Launch Complete Platform (Recommended)
```bash
# Navigate to the complete directory
cd complete

# Launch complete platform with UI and MLflow
docker compose -f compose-with-ui.yml up -d
```

This will start:
- **3 Docker container clients** (SuperExec-ClientApp-1, 2, 3)
- **1 Docker container server** (SuperExec-ServerApp)
- **SuperLink coordination service**
- **Real-time monitoring UI** at http://localhost:8050
- **MLflow tracking server** at http://localhost:5000

### Access the Platform
- **📊 Platform Dashboard**: http://localhost:8050
- **🌐 SuperLink API**: http://localhost:9093
- **📈 MLflow Tracking**: http://localhost:5000

### Run Federated Learning
```bash
# Start federated learning with container clients
cd complete
flwr run fl local-deployment --stream
```

### View Results
1. **Terminal**: Watch training progress and metrics
2. **Platform UI** (http://localhost:8050): Monitor container status and resources
3. **MLflow** (http://localhost:5000): Analyze training metrics, compare experiments
   - Click "Experiments" → "fl" to see all runs
   - View server and client-0, client-1, client-2 runs
   - Compare `train_loss` across clients and rounds

## 🐳 Docker Container Architecture

The platform runs **each client as a separate Docker container**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Federated Learning Platform                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   SuperLink     │    │   SuperNode-1   │    │ SuperNode-2  │ │
│  │   (Port 9093)   │◄──►│   (Port 9094)   │    │ (Port 9095)  │ │
│  │  Coordination   │    │   Client Node   │    │ Client Node  │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│           │                       │                       │     │
│           ▼                       ▼                       ▼     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │ SuperExec       │    │ SuperExec       │    │ SuperExec    │ │
│  │ ServerApp       │    │ ClientApp-1     │    │ ClientApp-2  │ │
│  │ (Docker)        │    │ (Docker)        │    │ (Docker)     │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │ SuperNode-3     │    │ Platform UI     │    │ MLflow       │ │
│  │ (Port 9096)     │    │ (Port 8050)     │    │ (Port 5000)  │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│           │                       │                       │     │
│           ▼                       │                       │     │
│  ┌─────────────────┐              │                       │     │
│  │ SuperExec       │              │                       │     │
│  │ ClientApp-3     │              │                       │     │
│  │ (Docker)        │              │                       │     │
│  └─────────────────┘              │                       │     │
└─────────────────────────────────────────────────────────────────┘
```

### Container Details
- **SuperLink**: Central coordination service
- **SuperExec-ServerApp**: Server running in Docker container
- **SuperNode-1, 2, 3**: Client nodes (ports 9094, 9095, 9096)
- **SuperExec-ClientApp-1, 2, 3**: Client containers with federated learning code
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

### Using Makefile Commands
For convenience, you can also use the provided Makefile commands:

**Platform Commands:**
```bash
make platform        # Launch complete platform with UI
make platform-stop   # Stop the platform
make platform-logs   # View platform logs
make showcase        # Start platform showcase demo
```

**Docker Commands:**
```bash
make docker-setup    # Setup Docker Compose environment
make docker-up       # Start services
make docker-test     # Run full test with services
make docker-down     # Stop services
make docker-logs     # View service logs
make docker-verify   # Verify Docker Compose setup
make docker-state    # Start with state persistence
make docker-tls      # Start with TLS encryption
make docker-certs    # Generate TLS certificates
make docker-combined # Start with state + TLS
make docker-clients  # Run with Docker container clients
make docker-add-clients # Add more client containers
```

### Alternative: Local Simulation
For local development and testing:
```bash
cd complete/fl
pip install -e .
flwr run .
```

See `complete/fl/README.md` for detailed instructions and advanced configuration.

## 📚 Documentation

### 🎯 Getting Started Guides

- **[SUMMARY.md](SUMMARY.md)** - 📋 **START HERE** - Quick overview and verification checklist
- **[RUNNING_GUIDE.md](RUNNING_GUIDE.md)** - 📖 Complete step-by-step instructions
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - ⚡ Quick command reference
- **[MLFLOW_GUIDE.md](MLFLOW_GUIDE.md)** - 📊 How to use MLflow for experiment tracking

### 📘 Advanced Guides

- **[Platform Showcase Guide](PLATFORM_SHOWCASE.md)** - Complete platform demonstration
- **[Docker Container Clients Guide](DOCKER_CLIENTS_GUIDE.md)** - Docker container clients setup
- **[Docker Compose Setup](DOCKER_COMPOSE.md)** - Detailed Docker Compose configuration
- **[Flower App Documentation](complete/fl/README.md)** - Application-specific documentation
- **[Official Flower Tutorial](https://flower.ai/docs/framework/docker/tutorial-quickstart-docker-compose.html)** - Authoritative reference

## 🎯 Platform Showcase

This platform demonstrates:

1. **Production-Ready Architecture**: Each client runs in a separate Docker container
2. **Real-time Monitoring**: Web UI showing container status, system resources, and federated learning progress
3. **Experiment Tracking**: MLflow integration for complete experiment reproducibility
4. **Scalable Design**: Easy to add more client containers
5. **Complete Stack**: From data loading to model training to experiment tracking
6. **Security Features**: TLS encryption and state persistence options

## 🚀 Getting Started

1. **Launch the platform**: `./launch-platform.sh`
2. **Open the dashboard**: http://localhost:8050
3. **Start federated learning**: `flwr run quickstart-compose local-deployment --stream`
4. **Monitor progress**: Watch real-time updates in the web UI

Perfect for showcasing federated learning with Docker container clients! 🎉
