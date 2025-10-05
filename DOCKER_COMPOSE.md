# Docker Compose Setup for Flower Framework

This document provides detailed information about running the federated learning project using Docker Compose with the Flower framework.

## Overview

The project uses the Flower framework's Docker Compose setup to run federated learning experiments in a distributed environment. This setup includes:

- **SuperLink**: Central coordination service
- **SuperNodes**: Client nodes that participate in federated learning
- **SuperExec**: Execution environments for server and client applications

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   SuperLink     │    │   SuperNode-1   │    │   SuperNode-2   │
│   (Port 9093)   │◄──►│   (Port 9094)   │    │   (Port 9095)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ SuperExec       │    │ SuperExec       │    │ SuperExec       │
│ (ServerApp)     │    │ (ClientApp-1)   │    │ (ClientApp-2)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

1. **Docker**: Install Docker Engine (version 20.10 or later)
2. **Docker Compose**: Install Docker Compose V2
3. **flwr CLI**: Install the Flower CLI tool
   ```bash
   pip install flwr
   ```

## Quick Start

### Automated Setup
```bash
./setup-docker-compose.sh
```

### Manual Setup
1. Clone the Docker Compose files:
   ```bash
   git clone --depth=1 --branch v1.22.0 https://github.com/adap/flower.git _tmp \
     && mv _tmp/framework/docker/complete . \
     && rm -rf _tmp
   ```

2. Create a Flower project:
   ```bash
   flwr new quickstart-compose --framework PyTorch --username flower
   ```

3. Set environment variable:
   ```bash
   export PROJECT_DIR=quickstart-compose
   ```

## Running the Project

### Start Services
```bash
docker compose up --build -d
```

### Run Federated Learning
```bash
flwr run quickstart-compose local-deployment --stream
```

### Stop Services
```bash
docker compose down
```

## Configuration Files

### compose.yml
Main Docker Compose configuration with:
- SuperLink service (port 9093)
- Two SuperNode services (ports 9094, 9095)
- SuperExec services for server and client applications
- Volume mounts for configuration and state persistence

### with-state.yml
Configuration for persisting SuperLink state:
- Enables database storage at `state/state.db`
- Mounts state directory for persistence

### with-tls.yml
Configuration for TLS encryption:
- Enables SSL/TLS for secure communication
- Uses certificate-based authentication
- Requires certificate generation via `certs.yml`

### certs.yml
Certificate generation service:
- Creates SSL certificates for TLS setup
- Generates CA and server certificates
- Stores certificates in `superlink-certificates/` directory

## Environment Variables

- `PROJECT_DIR`: Path to the Flower project directory (default: `quickstart-compose`)
- `FLWR_VERSION`: Flower framework version (default: `1.22.0`)

## Ports

- **9093**: SuperLink API and health checks
- **9094**: SuperNode-1 client API
- **9095**: SuperNode-2 client API
- **9096**: SuperNode-3 client API (if enabled)

## Volumes

- `superlink-state`: Persistent storage for SuperLink state
- `./fl/config/default.yaml`: Configuration file mount
- `./state/`: State persistence directory
- `./superlink-certificates/`: TLS certificates directory

## Makefile Commands

```bash
make docker-setup    # Setup Docker Compose environment
make docker-up       # Start services
make docker-test     # Run full test with services
make docker-down     # Stop services
make docker-logs     # View service logs
```

## Troubleshooting

### Services Not Starting
1. Check Docker is running: `docker info`
2. Check port availability: `netstat -tulpn | grep :9093`
3. View logs: `docker compose logs`

### Connection Issues
1. Verify SuperLink is accessible: `curl http://localhost:9093/health`
2. Check service status: `docker compose ps`
3. Restart services: `docker compose restart`

### Certificate Issues (TLS)
1. Generate certificates: `docker compose -f certs.yml up`
2. Verify certificate files exist in `superlink-certificates/`
3. Use `with-tls.yml` configuration

## Security Considerations

### Development (Insecure)
- Uses `--insecure` flag for development
- No TLS encryption
- Suitable for local testing only

### Production (Secure)
- Enable TLS using `with-tls.yml`
- Generate proper certificates
- Remove `--insecure` flags
- Use proper authentication

## Monitoring

### Health Checks
```bash
# Check SuperLink health
curl http://localhost:9093/health

# Check service status
docker compose ps

# View logs
docker compose logs -f
```

### Resource Usage
```bash
# View resource usage
docker stats

# View service details
docker compose top
```

## Advanced Configuration

### Adding More SuperNodes
Uncomment the SuperNode-3 section in `compose.yml` and update partition configurations.

### Custom Configuration
Modify `complete/fl/config/default.yaml` for experiment-specific settings.

### Persistent State
Use `with-state.yml` to enable state persistence across restarts.

## Integration with CI/CD

The project includes GitHub Actions workflow that:
1. Sets up Docker Compose environment
2. Builds and starts services
3. Runs basic connectivity tests
4. Cleans up resources

See `.github/workflows/ci.yml` for details.

## References

- [Flower Framework Documentation](https://flower.ai/docs/)
- [Official Docker Compose Quickstart Tutorial](https://flower.ai/docs/framework/docker/tutorial-quickstart-docker-compose.html) - **This is the authoritative guide**
- [Flower Deployment Engine](https://flower.ai/docs/framework/docker/)

## Official Tutorial Steps

This implementation follows the official Flower tutorial exactly:

1. **Step 1**: Set up the environment (automated via `setup-docker-compose.sh`)
2. **Step 2**: Run Flower in insecure mode (`docker compose up --build -d`)
3. **Step 3**: Run the quickstart project (`flwr run quickstart-compose local-deployment --stream`)
4. **Step 4**: Update the application (modify code and rebuild)
5. **Step 5**: Persist SuperLink state (`docker compose -f compose.yml -f with-state.yml up --build -d`)
6. **Step 6**: Run with TLS (`docker compose -f compose.yml -f with-tls.yml up --build -d`)
7. **Step 7**: Add more SuperNodes (uncomment in compose.yml)
8. **Step 8**: Combine state persistence and TLS
9. **Step 9**: Merge multiple compose files
10. **Step 10**: Clean up (`docker compose down -v`)

For the complete official tutorial, visit: [https://flower.ai/docs/framework/docker/tutorial-quickstart-docker-compose.html](https://flower.ai/docs/framework/docker/tutorial-quickstart-docker-compose.html)
