# Architecture

## System Overview

Production federated learning system with privacy, security, and communication efficiency.

```
Kubernetes Cluster
├── FL Server (SuperLink)
├── FL Clients (Workers)  
├── MLflow (Tracking)
└── Prometheus + Grafana (Monitoring)
```

## Core Components

### Data Loading

Thread-safe singleton pattern for dataset management.

```python
class FederatedDataLoader:
    _lock = threading.Lock()
    _instances: Dict[str, 'FederatedDataLoader'] = {}
```

Features:
- Lazy initialization
- Patient-based partitioning
- Configurable transforms
- 80/20 train/test split

### Gradient Compression

Hybrid compression combining quantization and sparsification.

Pipeline:
1. Add error feedback from previous round
2. Apply top-k sparsification (keep 10%)
3. Quantize to 8 bits
4. Store error for next round

Expected: 20-50x compression, <2% accuracy loss

### Byzantine-Robust Aggregation

Defense mechanisms:

**Multi-Krum**: Select k clients with smallest pairwise distances

**Trimmed Mean**: Sort element-wise, trim extremes, average

**Anomaly Detection**: Z-score based outlier detection

Handles up to 30% malicious clients

### Privacy Validation

Membership inference attacks for empirical privacy auditing.

Process:
1. Train shadow models
2. Extract prediction confidence
3. Train attack model
4. Evaluate on target

Metrics: Attack accuracy, AUC-ROC, privacy budget correlation

## Training Round Flow

```
1. Server broadcasts model
2. Clients train locally
3. Clients compress gradients
4. Clients send updates
5. Server detects Byzantine clients
6. Server aggregates (robust)
7. Server updates model
8. Repeat
```

## Deployment

### Kubernetes Resources

**Server**: 1 replica, 512Mi-1Gi RAM, 0.5-1 CPU

**Clients**: 5-20 replicas (HPA), 1-2Gi RAM, 1-2 CPU

**MLflow**: StatefulSet, 10Gi storage

**Monitoring**: Prometheus + Grafana

### Configuration

Helm chart with configurable:
- Number of clients
- Training rounds
- Learning rate
- Privacy budget
- Compression settings

## Security

1. Differential Privacy (DP-SGD)
2. Secure Aggregation (encrypted updates)
3. Byzantine Robustness (Multi-Krum, Trimmed Mean)
4. TLS/mTLS (encrypted communication)

## Monitoring

Metrics:
- Training: loss, accuracy, convergence
- Communication: bytes, compression ratio, bandwidth
- Security: Byzantine detection, privacy budget
- System: CPU, memory, latency

Dashboards:
1. Training Overview
2. Communication Efficiency
3. Byzantine Detection

## Best Practices

Development:
- Use virtual environments
- Run tests before committing
- Follow PEP 8
- Add type hints

Deployment:
- Use Helm
- Set resource limits
- Enable autoscaling
- Configure health checks

Security:
- Enable DP for sensitive data
- Monitor Byzantine behavior
- Rotate certificates
- Audit privacy regularly
