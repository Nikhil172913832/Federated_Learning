# Troubleshooting Guide

This guide helps resolve common issues with the Federated Learning platform.

## Table of Contents
- [Installation Issues](#installation-issues)
- [Docker Issues](#docker-issues)
- [Training Issues](#training-issues)
- [Data Loading Issues](#data-loading-issues)
- [Performance Issues](#performance-issues)
- [MLflow Issues](#mlflow-issues)

---

## Installation Issues

### Issue: `pip install -e ".[dev]"` fails

**Symptoms**: Error installing dependencies

**Solutions**:
```bash
# Update pip
python -m pip install --upgrade pip

# Install build tools
pip install setuptools wheel

# Try again
cd complete/fl
pip install -e ".[dev]"
```

### Issue: PyTorch installation fails

**Symptoms**: CUDA version mismatch or installation errors

**Solutions**:
```bash
# For CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Docker Issues

### Issue: Port already in use

**Symptoms**: 
```
Error: bind: address already in use
```

**Solutions**:
```bash
# Check what's using the port
sudo lsof -i :8050  # Dashboard
sudo lsof -i :5000  # MLflow
sudo lsof -i :9093  # SuperLink

# Kill the process
kill -9 <PID>

# Or change port in compose file
# Edit complete/compose-with-ui.yml
```

### Issue: Docker containers won't start

**Symptoms**: Containers exit immediately

**Solutions**:
```bash
# Check logs
docker logs complete-superlink-1
docker logs complete-superexec-serverapp-1

# Clean up and restart
cd complete
docker compose down -v
docker system prune -f
docker compose up --build -d

# Check status
docker compose ps
```

### Issue: Out of memory

**Symptoms**: Container killed, OOM errors

**Solutions**:
```bash
# Increase Docker memory limit (Docker Desktop)
# Settings → Resources → Memory → 8GB+

# Or reduce batch size in config
# Edit complete/fl/config/default.yaml
data:
  batch_size: 16  # Was 32

# Reduce number of clients
topology:
  num_clients: 5  # Was 10
```

---

## Training Issues

### Issue: Training fails immediately

**Symptoms**: Error on first round

**Diagnostic Steps**:
```bash
# Check logs
docker logs complete-superexec-clientapp-1-1

# Run locally for better error messages
cd complete/fl
python -c "from fl.task import load_data; load_data(0, 3)"
```

**Common Causes**:

1. **Invalid partition_id**:
```python
# Error: partition_id must be in [0, num_partitions)
# Fix: Check supernode configs match num_partitions
```

2. **Dataset download failed**:
```bash
# Manually download dataset
python -c "from datasets import load_dataset; load_dataset('albertvillanova/medmnist-v2', 'pneumoniamnist')"
```

3. **CUDA out of memory**:
```yaml
# Use CPU instead
# In Dockerfile, use Dockerfile.cpu
```

### Issue: Training is very slow

**Symptoms**: Each round takes >5 minutes

**Solutions**:
```bash
# Profile training
cd complete/fl
python scripts/profile_training.py --config config/default.yaml --output profiling/

# Check bottlenecks in profiling/trace_*.json
# Open in Chrome: chrome://tracing

# Common fixes:
# 1. Use GPU if available
# 2. Increase batch size
# 3. Reduce data augmentation
# 4. Use fewer local epochs
```

### Issue: Accuracy not improving

**Symptoms**: Accuracy stuck at ~10% (random guessing)

**Diagnostic Steps**:
```python
# Check data distribution
from fl.data_validation import DataValidator
validator = DataValidator()
report = validator.validate_partition(0, trainloader)
print(report.class_distribution)
```

**Common Causes**:

1. **Learning rate too high/low**:
```yaml
train:
  lr: 0.01  # Try 0.001 or 0.1
```

2. **Data not normalized**:
```python
# Check pixel values
images = next(iter(trainloader))["image"]
print(f"Min: {images.min()}, Max: {images.max()}")
# Should be roughly [-3, 3] after normalization
```

3. **Model architecture mismatch**:
```python
# Check input shape matches model
# For 28x28 images, model expects (1, 28, 28)
```

---

## Data Loading Issues

### Issue: `ValueError: partition_id must be in [0, num_partitions)`

**Symptoms**: Data loading fails with partition error

**Solution**:
```python
# Check partition configuration
# In compose.yml, ensure partition-id < num-partitions
environment:
  - PARTITION_ID=0  # Must be < num_partitions
```

### Issue: Dataset download hangs

**Symptoms**: Stuck downloading dataset

**Solutions**:
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache

# Download manually
python -c "
from datasets import load_dataset
ds = load_dataset('albertvillanova/medmnist-v2', 'pneumoniamnist', trust_remote_code=True)
print('Downloaded successfully')
"

# Check disk space
df -h
```

### Issue: Class imbalance warning

**Symptoms**: `⚠️ High class imbalance`

**This is expected** for non-IID data. To verify:
```python
from fl.data_validation import DataValidator
validator = DataValidator()
report = validator.validate_partition(0, trainloader, set_baseline=True)
print(f"Class distribution: {report.class_distribution}")
```

---

## Performance Issues

### Issue: High memory usage

**Symptoms**: System running out of RAM

**Solutions**:
```bash
# Monitor memory
docker stats

# Reduce batch size
# Edit config/default.yaml
data:
  batch_size: 16

# Reduce number of clients
topology:
  num_clients: 5

# Use gradient accumulation instead of large batches
```

### Issue: Slow network communication

**Symptoms**: Long delays between rounds

**Solutions**:
```bash
# Check network
docker network inspect complete_default

# Use local deployment instead of Docker
cd complete/fl
flwr run . local-simulation --stream

# Reduce model size
# Use smaller architecture in fl/task.py
```

---

## MLflow Issues

### Issue: MLflow UI not accessible

**Symptoms**: Cannot access http://localhost:5000

**Solutions**:
```bash
# Check MLflow container
docker logs complete-mlflow-1

# Restart MLflow
docker restart complete-mlflow-1

# Check port mapping
docker port complete-mlflow-1

# Access via container IP
docker inspect complete-mlflow-1 | grep IPAddress
```

### Issue: Metrics not logging

**Symptoms**: No runs in MLflow UI

**Solutions**:
```python
# Check MLflow connection
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
print(mlflow.get_tracking_uri())

# Verify logging works
with mlflow.start_run():
    mlflow.log_param("test", 1)
    mlflow.log_metric("test_metric", 0.5)
```

### Issue: Experiment not found

**Symptoms**: `Experiment 'fl' not found`

**Solution**:
```python
# Create experiment manually
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.create_experiment("fl")
```

---

## Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Run Components Locally

```bash
# Test data loading
cd complete/fl
python -c "from fl.task import load_data; trainloader, _ = load_data(0, 3); print(f'Loaded {len(trainloader.dataset)} samples')"

# Test training
python -c "
from fl.task import Net, load_data, train
import torch
model = Net()
trainloader, _ = load_data(0, 3)
device = torch.device('cpu')
loss = train(model, trainloader, 1, 0.01, device)
print(f'Training loss: {loss}')
"
```

### Check Configuration

```bash
# Validate config
cd complete/fl
python -c "
from fl.config_schema import validate_config_file
config = validate_config_file('config/default.yaml')
print('Config valid!')
"
```

### Profile Performance

```bash
cd complete/fl
python scripts/profile_training.py \
    --config config/default.yaml \
    --output profiling/ \
    --epochs 1
```

---

## Getting Help

If you're still stuck:

1. **Check logs**: `docker compose logs -f`
2. **Search issues**: [GitHub Issues](https://github.com/Nikhil172913832/Federated_Learning/issues)
3. **Ask for help**: [GitHub Discussions](https://github.com/Nikhil172913832/Federated_Learning/discussions)
4. **Provide details**:
   - Error messages
   - Steps to reproduce
   - Environment (OS, Python version, Docker version)
   - Config file
   - Logs
