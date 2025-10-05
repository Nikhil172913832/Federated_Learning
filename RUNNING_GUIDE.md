# 🚀 Complete Guide: Running the Federated Learning Platform

This guide will help you run the complete federated learning platform with **Docker container clients**, **real-time UI monitoring**, and **MLflow experiment tracking**.

## 📋 Prerequisites

Before starting, ensure you have:

- ✅ **Docker** and **Docker Compose V2** installed and running
- ✅ **Python 3.8+** installed
- ✅ **Flower CLI** installed (`pip install flwr`)

## 🎯 Quick Start (Complete Platform with UI & MLflow)

### Step 1: Stop Any Running Containers

```bash
cd /home/darklord/Projects/Federated_Learning/complete
docker compose down
```

### Step 2: Start the Complete Platform

```bash
# Use absolute path to avoid directory issues
docker compose -f /home/darklord/Projects/Federated_Learning/complete/compose-with-ui.yml up -d --build
```

This will start:
- **3 Docker container clients** (SuperExec-ClientApp-1, 2, 3)
- **1 Docker container server** (SuperExec-ServerApp)
- **SuperLink** - Central coordination service (port 9093)
- **3 SuperNodes** - Client node managers (ports 9094, 9095, 9096)
- **Platform UI** - Real-time monitoring dashboard (port 8050)
- **MLflow** - Experiment tracking server (port 5000)

### Step 3: Verify All Services Are Running

```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

You should see:
- `complete-fl-platform-ui-1` → Port 8050
- `complete-mlflow-1` → Port 5000
- `complete-superlink-1` → Port 9093
- `complete-supernode-1-1`, `complete-supernode-2-1`, `complete-supernode-3-1`
- `complete-superexec-serverapp-1`
- `complete-superexec-clientapp-1-1`, `complete-superexec-clientapp-2-1`, `complete-superexec-clientapp-3-1`

### Step 4: Access the Monitoring Dashboards

Open these URLs in your browser:

- **📊 Platform UI**: http://localhost:8050
  - Real-time container status
  - System resource monitoring
  - Training progress visualization

- **📈 MLflow Tracking**: http://localhost:5000
  - Experiment tracking
  - Model versioning
  - Training metrics and parameters

- **🌐 SuperLink API**: http://localhost:9093
  - API endpoint for federated learning coordination

### Step 5: Run Federated Learning

```bash
cd /home/darklord/Projects/Federated_Learning/complete
flwr run fl local-deployment --stream
```

This command will:
1. Connect to the SuperLink at `127.0.0.1:9093`
2. Distribute the ServerApp and ClientApps to the containers
3. Execute 3 rounds of federated learning
4. Each round:
   - Server sends global model to clients
   - Clients train on their local data partitions
   - Clients send updates back to server
   - Server aggregates updates using FedAvg
5. Save the final model as `final_model.pt`

### Step 6: Monitor the Training

**In the terminal**, you'll see:
```
Loading project configuration from pyproject.toml
Success
Run Flower app
```

**In the Platform UI (http://localhost:8050)**, you'll see:
- Container status and health
- System resource usage (CPU, memory)
- Real-time updates during training

**In MLflow (http://localhost:5000)**, you'll see:
- Experiment: `fl`
- Runs for `server` and each `client-0`, `client-1`, `client-2`
- Metrics: `train_loss` for each client per round
- Parameters: learning rate, epochs, partition IDs, etc.

## 📊 Understanding the Output

### Terminal Output
```
Round 1:
- Client 0: Training with partition 0/3...
- Client 1: Training with partition 1/3...
- Client 2: Training with partition 2/3...
- Server: Aggregating updates...

Round 2:
- [Same process repeats]

Round 3:
- [Final round]

Saving final model to disk...
Model saved! Training completed with 3 rounds.
```

### MLflow Metrics

Navigate to http://localhost:5000 and click on the `fl` experiment. You'll see:

**Server Run:**
- Parameters: `num_rounds=3`, `fraction_train=0.5`, `lr=0.01`
- Metrics: `training_complete=1`

**Client Runs (client-0, client-1, client-2):**
- Parameters: `partition_id`, `local_epochs`, `lr`
- Metrics: `train_loss` (tracked per round)

## 🛠️ Configuration

### Dataset & Training Parameters
Edit `complete/fl/config/default.yaml`:

```yaml
train:
  lr: 0.01              # Learning rate
  local_epochs: 1       # Epochs per client per round
  num_server_rounds: 3  # Total federated rounds

topology:
  num_clients: 10       # Total clients (3 in Docker mode)
  fraction: 0.5         # Fraction of clients per round

data:
  dataset: albertvillanova/medmnist-v2
  subset: pneumoniamnist
  iid: true             # IID data distribution
```

### Change Number of Rounds
In `complete/fl/pyproject.toml`:
```toml
[tool.flwr.app.config]
num-server-rounds = 5  # Change this
```

## 🔧 Troubleshooting

### Platform UI Not Loading
```bash
# Check UI logs
docker logs complete-fl-platform-ui-1

# Restart UI container
docker restart complete-fl-platform-ui-1
```

### MLflow Not Accessible
```bash
# Check MLflow logs
docker logs complete-mlflow-1

# Ensure port 5000 is not in use
lsof -i :5000
```

### Training Not Starting
```bash
# Check SuperLink logs
docker logs complete-superlink-1

# Check if flwr CLI is installed
which flwr || pip install flwr

# Verify containers are running
docker ps | grep complete
```

### Import Errors
If you see import errors like `RandomFixedSizePartitioner`:
- The codebase has been updated to use `ShardPartitioner`
- Rebuild containers: `docker compose -f compose-with-ui.yml up -d --build`

## 📁 Output Files

After training completes, you'll find:

- **`final_model.pt`** - Trained global model (PyTorch state dict)
- **MLflow artifacts** - Stored in the `mlflow-data` Docker volume
- **Client checkpoints** - Stored during training (if configured)

## 🧹 Cleanup

### Stop Platform (Keep Data)
```bash
docker compose -f /home/darklord/Projects/Federated_Learning/complete/compose-with-ui.yml down
```

### Stop Platform (Remove Data)
```bash
docker compose -f /home/darklord/Projects/Federated_Learning/complete/compose-with-ui.yml down -v
```

### Remove All Images
```bash
docker compose -f /home/darklord/Projects/Federated_Learning/complete/compose-with-ui.yml down --rmi all -v
```

## 🎓 Advanced Usage

### Run with State Persistence
```bash
docker compose -f compose-with-ui.yml -f with-state.yml up -d
```

### Run with TLS Encryption
```bash
# Generate certificates first
docker compose -f certs.yml run --rm --build gen-certs

# Start with TLS
docker compose -f compose-with-ui.yml -f with-tls.yml up -d
```

### View Real-time Logs
```bash
# All containers
docker compose -f compose-with-ui.yml logs -f

# Specific service
docker logs -f complete-superexec-serverapp-1
docker logs -f complete-superexec-clientapp-1-1
docker logs -f complete-mlflow-1
```

### Scale Clients
To add more clients, edit `compose-with-ui.yml` and add more `supernode-N` and `superexec-clientapp-N` services.

## 📚 Key Concepts

### Federated Learning Flow
1. **Initialization**: Server creates global model
2. **Distribution**: Global model sent to selected clients
3. **Local Training**: Each client trains on local data
4. **Aggregation**: Server averages client updates (FedAvg)
5. **Iteration**: Repeat for multiple rounds
6. **Finalization**: Save final global model

### Data Partitioning
- Dataset: MedMNIST (Pneumonia detection)
- 3 clients, each gets a partition of the data
- IID partitioning (configurable to non-IID)

### MLflow Integration
- Automatic experiment tracking
- Metrics logged per client per round
- Parameters logged for reproducibility
- Artifacts saved for model versioning

## 🎉 Success!

If everything is working, you should see:
- ✅ All containers running
- ✅ Platform UI showing container status
- ✅ MLflow showing experiment runs and metrics
- ✅ Training completing successfully
- ✅ `final_model.pt` saved

Congratulations! You've successfully run a production-ready federated learning platform with Docker containers, real-time monitoring, and experiment tracking! 🚀

---

## 📞 Need Help?

- Check the logs: `docker compose logs -f`
- Verify network: `docker network ls`
- Restart services: `docker compose restart`
- Rebuild from scratch: `docker compose down -v && docker compose -f compose-with-ui.yml up -d --build`
