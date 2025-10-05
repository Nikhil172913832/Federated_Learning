# 🎯 Quick Reference: Federated Learning Platform

## ⚡ Quick Commands

### Start Platform with UI & MLflow
```bash
cd /home/darklord/Projects/Federated_Learning/complete
docker compose -f compose-with-ui.yml up -d
```

### Run Federated Learning
```bash
cd /home/darklord/Projects/Federated_Learning/complete
flwr run fl local-deployment --stream
```

### Access Dashboards
- **Platform UI**: http://localhost:8050
- **MLflow**: http://localhost:5000
- **SuperLink API**: http://localhost:9093

### View Logs
```bash
# All containers
docker compose -f compose-with-ui.yml logs -f

# Server only
docker logs -f complete-superexec-serverapp-1

# Client 1 only
docker logs -f complete-superexec-clientapp-1-1

# MLflow only
docker logs -f complete-mlflow-1

# Platform UI only
docker logs -f complete-fl-platform-ui-1
```

### Stop Platform
```bash
docker compose -f compose-with-ui.yml down
```

### Rebuild After Code Changes
```bash
docker compose -f compose-with-ui.yml down
docker compose -f compose-with-ui.yml up -d --build
```

### Check Container Status
```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

## 📊 What Gets Logged to MLflow

### Server Experiment
- **Experiment Name**: `fl`
- **Run Name**: `server`
- **Parameters**:
  - `num_rounds`: Number of federated rounds (default: 3)
  - `fraction_train`: Fraction of clients per round (default: 0.5)
  - `lr`: Learning rate (default: 0.01)
  - `num_partitions`: Total number of client partitions (3)
- **Metrics**:
  - `training_complete`: Set to 1 when training finishes

### Client Experiments
- **Experiment Name**: `fl`
- **Run Names**: `client-0`, `client-1`, `client-2`
- **Parameters** (per client):
  - `partition_id`: Which data partition (0, 1, or 2)
  - `num_partitions`: Total partitions (3)
  - `local_epochs`: Training epochs per round (default: 1)
  - `lr`: Learning rate (default: 0.01)
- **Metrics** (per client, per round):
  - `train_loss`: Training loss after local training
  - Step number corresponds to the round number

## 🔍 Monitoring Training Progress

### 1. Terminal Output
Watch the `flwr run` command output for:
- Round numbers
- Client selection
- Training progress
- Aggregation status

### 2. Platform UI (http://localhost:8050)
- Container health status
- Resource utilization (CPU, Memory)
- Real-time system monitoring
- Container uptime

### 3. MLflow UI (http://localhost:5000)
Navigate to:
1. Click "Experiments" → "fl"
2. You'll see multiple runs (1 server + 3 clients)
3. Click on any run to see:
   - Parameters used
   - Metrics over time
   - Training loss charts
4. Compare runs to see performance across clients

### 4. Docker Logs
```bash
# Watch all logs in real-time
docker compose -f compose-with-ui.yml logs -f

# Filter for specific information
docker compose logs -f | grep -i "round\|loss\|accuracy"
```

## 📁 Output Files

After running federated learning:
- **`final_model.pt`** - Final global model (PyTorch state dict)
- **MLflow data** - Stored in Docker volume `mlflow-data`
- **SuperLink state** - Stored in Docker volume `superlink-state`

To access the model:
```bash
# The model is saved inside the server container
docker exec complete-superexec-serverapp-1 ls -lh /app/final_model.pt

# Copy it to your host machine
docker cp complete-superexec-serverapp-1:/app/final_model.pt ./final_model.pt
```

## 🛠️ Troubleshooting

### "flwr: command not found"
```bash
pip install flwr
# or
pip3 install flwr
```

### Platform UI not loading
```bash
# Check if port 8050 is in use
lsof -i :8050

# Restart the UI container
docker restart complete-fl-platform-ui-1
```

### MLflow not accessible
```bash
# Check MLflow logs
docker logs complete-mlflow-1

# Restart MLflow
docker restart complete-mlflow-1
```

### No metrics in MLflow
1. Make sure `MLFLOW_TRACKING_URI` environment variable is set in containers
2. Check that training actually ran successfully
3. Look for errors in client/server logs
4. The metrics appear after each round completes

### Training hangs or fails
```bash
# Check SuperLink
docker logs complete-superlink-1

# Check server
docker logs complete-superexec-serverapp-1

# Check clients
docker logs complete-superexec-clientapp-1-1
docker logs complete-superexec-clientapp-2-1
docker logs complete-superexec-clientapp-3-1

# Restart everything
docker compose -f compose-with-ui.yml restart
```

## ⚙️ Configuration

### Change Number of Training Rounds
Edit `complete/fl/config/default.yaml`:
```yaml
train:
  num_server_rounds: 5  # Change from 3 to 5
```

Or edit `complete/fl/pyproject.toml`:
```toml
[tool.flwr.app.config]
num-server-rounds = 5
```

### Change Learning Rate
Edit `complete/fl/config/default.yaml`:
```yaml
train:
  lr: 0.001  # Change from 0.01
```

### Change Local Epochs
Edit `complete/fl/config/default.yaml`:
```yaml
train:
  local_epochs: 2  # Change from 1
```

After changing configuration:
```bash
# Rebuild containers to pick up changes
docker compose -f compose-with-ui.yml down
docker compose -f compose-with-ui.yml up -d --build

# Run training again
flwr run fl local-deployment --stream
```

## 📈 Understanding the Results

### Expected Behavior
1. **Round 1**: Higher train_loss (model just starting)
2. **Round 2**: Lower train_loss (model improving)
3. **Round 3**: Lowest train_loss (model converging)

### In MLflow
- Click on "Charts" to visualize metrics
- Compare `train_loss` across different clients
- Download metrics as CSV for further analysis

### Comparing Runs
- In MLflow, select multiple runs (checkbox)
- Click "Compare" button
- View side-by-side parameter and metric comparison

## 🎓 Next Steps

### Experiment Ideas
1. **Increase rounds**: See how accuracy improves
2. **Change learning rate**: Compare convergence speed
3. **Non-IID data**: Edit config to use `non_iid` settings
4. **Differential Privacy**: Enable DP-SGD in config
5. **Personalization**: Try FedProx algorithm

### Production Deployment
- Add more clients (edit `compose-with-ui.yml`)
- Enable TLS encryption (`with-tls.yml`)
- Enable state persistence (`with-state.yml`)
- Set up proper monitoring and alerting
- Use production-grade MLflow backend

## 🎉 Success Indicators

You've successfully run the platform if:
- ✅ All 10 containers are running
- ✅ Platform UI shows green status at http://localhost:8050
- ✅ MLflow shows experiment at http://localhost:5000
- ✅ Training completes with "Model saved!" message
- ✅ You can see metrics in MLflow for each client
- ✅ `final_model.pt` exists in the server container

---

**For detailed instructions, see: [RUNNING_GUIDE.md](RUNNING_GUIDE.md)**
