# Docker Container Clients Guide

This guide explains how to run federated learning with **real Docker containers** as clients, following the official Flower framework documentation.

## Architecture Overview

When you run federated learning with Docker Compose, you get a **distributed system** with separate containers for each component:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Compose Network                       │
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
│  │ (Your Server)   │    │ (Your Client)   │    │ (Your Client)│ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Container Types

### 1. **SuperLink Container**
- **Purpose**: Central coordination service
- **Image**: `flwr/superlink:1.22.0`
- **Port**: 9093 (exposed to host)
- **Role**: Manages federated learning rounds, aggregates results

### 2. **SuperNode Containers**
- **Purpose**: Client nodes that host client applications
- **Image**: `flwr/supernode:1.22.0`
- **Ports**: 9094, 9095, 9096 (internal network)
- **Role**: Provide execution environment for client apps

### 3. **SuperExec-ServerApp Container**
- **Purpose**: Runs your server code in isolation
- **Image**: Built from your project (`complete/fl/`)
- **Role**: Executes server-side federated learning logic

### 4. **SuperExec-ClientApp Containers**
- **Purpose**: Run your client code in separate containers
- **Image**: Built from your project (`complete/fl/`)
- **Role**: Execute client-side federated learning logic
- **Isolation**: Each client runs in its own container with separate resources

## Key Benefits of Container Clients

### ✅ **True Isolation**
- Each client runs in its own Docker container
- Separate filesystem, network, and process space
- Simulates real-world distributed federated learning

### ✅ **Resource Management**
- CPU limits per client container
- Memory isolation
- Network isolation between clients

### ✅ **Scalability**
- Easy to add more client containers
- Can run on different machines
- Production-ready architecture

### ✅ **Realistic Simulation**
- Mimics actual federated learning deployment
- Tests network communication
- Validates distributed system behavior

## How to Run

### Quick Start
```bash
./run-docker-clients.sh
```

### Manual Steps
```bash
# 1. Setup (if not done already)
./setup-docker-compose.sh

# 2. Start all containers
cd complete
export PROJECT_DIR=quickstart-compose
docker compose up --build -d

# 3. Wait for services to be ready
sleep 30

# 4. Run federated learning
cd ..
flwr run quickstart-compose local-deployment --stream
```

## Monitoring Container Clients

### View All Containers
```bash
docker compose ps
```

### View Client Container Logs
```bash
# Server container logs
docker compose logs superexec-serverapp

# Client container logs
docker compose logs superexec-clientapp-1
docker compose logs superexec-clientapp-2

# All logs
docker compose logs -f
```

### Monitor Resource Usage
```bash
# Real-time resource usage
docker stats

# Container details
docker compose top
```

## Adding More Client Containers

### Automatic Addition
```bash
./add-more-clients.sh
```

### Manual Addition
1. Edit `complete/compose.yml`
2. Uncomment `supernode-3` and `superexec-clientapp-3` sections
3. Update partition configuration
4. Restart services:
   ```bash
   docker compose down
   docker compose up --build -d
   ```

## Container Configuration

### Client Container Resources
Each client container is configured with:
```yaml
deploy:
  resources:
    limits:
      cpus: "2"  # CPU limit per client
```

### Environment Variables
```yaml
environment:
  - FL_CONFIG_PATH=/app/config/default.yaml
```

### Volume Mounts
```yaml
volumes:
  - ./fl/config/default.yaml:/app/config/default.yaml:ro
```

## Network Architecture

### Internal Network
- All containers communicate via Docker's internal network
- SuperLink accessible at `superlink:9092` (internal)
- SuperNodes accessible at `supernode-1:9094`, `supernode-2:9095`

### External Access
- SuperLink exposed on `localhost:9093`
- Health checks available at `http://localhost:9093/health`

## Troubleshooting

### Containers Not Starting
```bash
# Check logs
docker compose logs

# Check resource usage
docker stats

# Restart services
docker compose restart
```

### Client Communication Issues
```bash
# Test SuperLink connectivity
curl http://localhost:9093/health

# Check network connectivity
docker compose exec supernode-1 ping superlink
```

### Resource Issues
```bash
# Check available resources
docker system df
docker system prune  # Clean up if needed
```

## Advanced Configuration

### Custom Client Configuration
Modify `complete/fl/config/default.yaml` to customize:
- Number of local epochs
- Learning rate
- Batch size
- Data partitioning

### TLS with Container Clients
```bash
# Generate certificates
docker compose -f certs.yml run --rm --build gen-certs

# Start with TLS
docker compose -f compose.yml -f with-tls.yml up --build -d
```

### State Persistence
```bash
# Start with state persistence
docker compose -f compose.yml -f with-state.yml up --build -d
```

## Comparison: Container vs Simulation

| Feature | Container Clients | Local Simulation |
|---------|------------------|------------------|
| **Isolation** | ✅ Full container isolation | ❌ Process isolation only |
| **Realism** | ✅ Production-like | ❌ Simplified |
| **Resource Management** | ✅ Per-container limits | ❌ Shared resources |
| **Scalability** | ✅ Easy to scale | ❌ Limited to single machine |
| **Network Testing** | ✅ Real network communication | ❌ In-memory communication |
| **Setup Complexity** | ⚠️ More complex | ✅ Simple |
| **Resource Usage** | ⚠️ Higher (multiple containers) | ✅ Lower |

## Best Practices

1. **Start Simple**: Begin with 2 client containers
2. **Monitor Resources**: Watch CPU and memory usage
3. **Check Logs**: Monitor container logs for issues
4. **Test Connectivity**: Verify SuperLink accessibility
5. **Scale Gradually**: Add more clients as needed
6. **Use TLS in Production**: Enable encryption for real deployments

## Next Steps

- [Official Flower Docker Compose Tutorial](https://flower.ai/docs/framework/docker/tutorial-quickstart-docker-compose.html)
- [Flower Deployment Engine Documentation](https://flower.ai/docs/framework/docker/)
- [Multi-Machine Deployment Guide](https://flower.ai/docs/framework/docker/tutorial-multi-machine-docker-compose.html)
