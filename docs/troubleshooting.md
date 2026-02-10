# Troubleshooting Guide

## Common Issues

### 1. Pods Not Starting

**Symptoms:**
- Pods stuck in `Pending` or `CrashLoopBackOff`
- `kubectl get pods` shows errors

**Diagnosis:**
```bash
kubectl describe pod <pod-name> -n fl-system
kubectl logs <pod-name> -n fl-system
```

**Common Causes:**

#### Insufficient Resources
```bash
# Check node resources
kubectl top nodes

# Solution: Scale down or add nodes
kubectl scale deployment fl-client --replicas=2
```

#### Image Pull Errors
```bash
# Check image
kubectl describe pod <pod-name> | grep -A 5 Events

# Solution: Verify image exists
docker images | grep fl-platform
```

#### Configuration Errors
```bash
# Check ConfigMap
kubectl get configmap fl-config -o yaml

# Solution: Update ConfigMap
kubectl edit configmap fl-config
kubectl rollout restart deployment/fl-server
```

---

### 2. Training Not Converging

**Symptoms:**
- Loss not decreasing
- Accuracy stuck at random

**Diagnosis:**
```bash
# Check training logs
kubectl logs deployment/fl-client -n fl-system | grep "loss"

# Check Grafana dashboard
kubectl port-forward svc/grafana 3000:3000
```

**Solutions:**

#### Learning Rate Too High/Low
```yaml
# Update values.yaml
config:
  learningRate: 0.001  # Try different values
```

#### Data Distribution Issues
```python
# Check partition statistics
from fl.datasets.nih_chest_xray import PatientBasedPartitioner
partitioner = PatientBasedPartitioner(metadata_path, num_hospitals=10)
stats = partitioner.get_partition_stats(partitions)
print(stats)
```

#### Compression Too Aggressive
```yaml
# Reduce compression
config:
  compression:
    sparsity: 0.5  # Less aggressive
    quantization_bits: 16  # More bits
```

---

### 3. High Memory Usage

**Symptoms:**
- OOMKilled pods
- Slow training

**Diagnosis:**
```bash
kubectl top pods -n fl-system
kubectl describe pod <pod-name> | grep -A 5 "Limits"
```

**Solutions:**

#### Increase Memory Limits
```yaml
# values.yaml
client:
  resources:
    limits:
      memory: "4Gi"  # Increase
```

#### Reduce Batch Size
```yaml
config:
  batchSize: 16  # Smaller batches
```

#### Enable Gradient Checkpointing
```python
# In model definition
model.gradient_checkpointing_enable()
```

---

### 4. Byzantine Detection False Positives

**Symptoms:**
- Too many clients flagged as Byzantine
- Training unstable

**Diagnosis:**
```bash
# Check Grafana Byzantine dashboard
# Look at anomaly scores
```

**Solutions:**

#### Adjust Detection Threshold
```python
# In robust_aggregation.py
byzantine = aggregator.detect_byzantine(
    client_updates,
    threshold=5.0  # Increase threshold
)
```

#### Use Different Aggregation Method
```yaml
config:
  security:
    aggregation_method: "multi_krum"  # Try different method
```

---

### 5. Privacy Budget Exhausted

**Symptoms:**
- DP training stops early
- Privacy errors

**Diagnosis:**
```bash
# Check privacy metrics
kubectl logs deployment/fl-client | grep "epsilon"
```

**Solutions:**

#### Increase Privacy Budget
```yaml
config:
  privacy:
    target_epsilon: 10.0  # Higher budget
```

#### Reduce Noise
```yaml
config:
  privacy:
    noise_multiplier: 0.5  # Less noise
```

#### Train for Fewer Rounds
```yaml
config:
  num_server_rounds: 5  # Fewer rounds
```

---

### 6. Slow Communication

**Symptoms:**
- Long round times
- High network latency

**Diagnosis:**
```bash
# Check network metrics
kubectl exec -it <pod-name> -- ping fl-server

# Check compression ratio
# View Grafana Communication dashboard
```

**Solutions:**

#### Increase Compression
```yaml
config:
  compression:
    sparsity: 0.95  # More aggressive
    quantization_bits: 4  # Fewer bits
```

#### Enable Async Aggregation
```python
# Implement async aggregation
# (Currently not implemented)
```

---

### 7. MLflow Not Accessible

**Symptoms:**
- Cannot access MLflow UI
- Experiments not logged

**Diagnosis:**
```bash
kubectl get svc mlflow -n fl-system
kubectl logs statefulset/mlflow -n fl-system
```

**Solutions:**

#### Port Forward
```bash
kubectl port-forward svc/mlflow 5000:5000 -n fl-system
```

#### Check Persistent Volume
```bash
kubectl get pvc -n fl-system
kubectl describe pvc mlflow-data-mlflow-0
```

#### Restart MLflow
```bash
kubectl rollout restart statefulset/mlflow -n fl-system
```

---

### 8. Grafana Dashboards Not Loading

**Symptoms:**
- Empty dashboards
- No data points

**Diagnosis:**
```bash
# Check Prometheus
kubectl port-forward svc/prometheus 9090:9090
# Visit http://localhost:9090

# Check metrics
curl http://localhost:9090/api/v1/query?query=fl_training_loss
```

**Solutions:**

#### Verify Metrics Export
```python
# In training code
from fl.metrics import metrics_collector
metrics_collector.record_training_loss(client_id, round_num, loss)
```

#### Check Prometheus Config
```bash
kubectl get configmap prometheus-config -o yaml
```

#### Restart Prometheus
```bash
kubectl rollout restart deployment/prometheus
```

---

## Performance Tuning

### Optimize Training Speed

1. **Use GPU**
```yaml
client:
  resources:
    limits:
      nvidia.com/gpu: 1
```

2. **Increase Workers**
```python
DataLoader(dataset, num_workers=4)
```

3. **Mixed Precision**
```python
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

### Optimize Communication

1. **Aggressive Compression**
```yaml
compression:
  sparsity: 0.99
  quantization_bits: 2
```

2. **Reduce Model Size**
```python
# Use smaller model
from fl.models import TinyNet
```

### Optimize Memory

1. **Gradient Accumulation**
```python
for i, batch in enumerate(dataloader):
    loss = train_step(batch)
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

2. **Clear Cache**
```python
import torch
torch.cuda.empty_cache()
```

---

## Debugging Commands

```bash
# Get all resources
kubectl get all -n fl-system

# Describe deployment
kubectl describe deployment fl-server -n fl-system

# View logs (last 100 lines)
kubectl logs --tail=100 deployment/fl-client -n fl-system

# Follow logs
kubectl logs -f deployment/fl-server -n fl-system

# Execute command in pod
kubectl exec -it <pod-name> -n fl-system -- bash

# Check events
kubectl get events -n fl-system --sort-by='.lastTimestamp'

# Resource usage
kubectl top pods -n fl-system
kubectl top nodes

# Port forward multiple services
kubectl port-forward svc/mlflow 5000:5000 &
kubectl port-forward svc/grafana 3000:3000 &
kubectl port-forward svc/prometheus 9090:9090 &
```

---

## Getting Help

1. **Check Logs First**
   - Server logs: `kubectl logs deployment/fl-server`
   - Client logs: `kubectl logs deployment/fl-client`
   - MLflow logs: `kubectl logs statefulset/mlflow`

2. **Check Metrics**
   - Grafana dashboards
   - Prometheus queries
   - MLflow experiments

3. **Check Configuration**
   - ConfigMaps
   - Secrets
   - Helm values

4. **GitHub Issues**
   - Search existing issues
   - Create new issue with logs and config
