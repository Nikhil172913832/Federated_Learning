# API Documentation

## Core Modules

### Data Loading

#### `FederatedDataLoader`

Thread-safe singleton for federated dataset management.

```python
from fl.data_loader import FederatedDataLoader
from fl.config import load_run_config

config = load_run_config()
loader = FederatedDataLoader.get_instance(config)

# Load partition for client
train_loader, test_loader = loader.load_partition(
    partition_id=0,
    num_partitions=10
)
```

**Methods:**
- `get_instance(config)`: Get singleton instance
- `load_partition(partition_id, num_partitions)`: Load client partition
- `get_dataset_info()`: Get dataset statistics

---

### Gradient Compression

#### `HybridCompressor`

Combines quantization and sparsification with error feedback.

```python
from fl.compression import HybridCompressor

compressor = HybridCompressor(
    quantization_bits=8,
    sparsity=0.9,
    use_error_feedback=True
)

# Compress gradients
compressed, metadata, stats = compressor.compress_gradients(state_dict)

# Decompress
decompressed = compressor.decompress_gradients(compressed, metadata)
```

**Parameters:**
- `quantization_bits` (int): Bits for quantization (1-32)
- `sparsity` (float): Fraction to zero out (0-1)
- `use_error_feedback` (bool): Enable error accumulation

**Returns:**
- `compressed`: Compressed state dict
- `metadata`: Decompression metadata
- `stats`: CompressionStats object

---

### Byzantine-Robust Aggregation

#### `RobustAggregator`

Ensemble of robust aggregation methods.

```python
from fl.robust_aggregation import RobustAggregator

aggregator = RobustAggregator(
    method="trimmed_mean",
    num_byzantine=2,
    trim_ratio=0.1
)

# Aggregate client updates
aggregated = aggregator.aggregate(client_updates)

# Detect Byzantine clients
byzantine_ids = aggregator.detect_byzantine(client_updates)
```

**Methods:**
- `multi_krum`: Select k closest gradients
- `trimmed_mean`: Trim extreme values
- `median`: Coordinate-wise median

**Parameters:**
- `method` (str): Aggregation method
- `num_byzantine` (int): Expected malicious clients
- `trim_ratio` (float): Fraction to trim (0-0.5)

---

### Privacy Auditing

#### `PrivacyAuditor`

Audit privacy via membership inference attacks.

```python
from fl.privacy.membership_inference import PrivacyAuditor

auditor = PrivacyAuditor()

# Audit model
result = auditor.audit_model(
    model=trained_model,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    epsilon=3.0
)

print(f"Attack accuracy: {result.attack_accuracy:.3f}")
print(f"AUC: {result.auc:.3f}")
```

**Returns:**
- `attack_accuracy`: MIA success rate
- `attack_precision`: Precision
- `attack_recall`: Recall
- `auc`: ROC-AUC score
- `epsilon`: Privacy budget

---

### Metrics Collection

#### `MetricsCollector`

Prometheus metrics instrumentation.

```python
from fl.metrics import metrics_collector

# Record training metrics
metrics_collector.record_training_loss(
    client_id="client_0",
    round_num=1,
    loss=0.5
)

# Record communication
metrics_collector.record_communication(
    client_id="client_0",
    sent_bytes=1024,
    received_bytes=512,
    original_size=10240,
    compressed_size=512,
    round_num=1
)

# Record Byzantine detection
metrics_collector.record_byzantine_detection(
    round_num=1,
    count=2
)
```

---

## Configuration

### YAML Configuration

```yaml
seed: 42

topology:
  num_clients: 10
  fraction: 0.5

train:
  lr: 0.01
  local-epochs: 1
  num-server-rounds: 10

data:
  dataset: "albertvillanova/medmnist-v2"
  subset: "pneumoniamnist"
  batch_size: 32

privacy:
  dp_sgd:
    enabled: true
    noise_multiplier: 0.8
    target_epsilon: 3.0

compression:
  enabled: true
  quantization_bits: 8
  sparsity: 0.9

security:
  aggregation_method: "trimmed_mean"
  byzantine_ratio: 0.2
```

---

## CLI Usage

### Training

```bash
# Local simulation
flwr run . local-simulation --stream

# With custom config
flwr run . local-simulation \
  --run-config num-server-rounds=50 \
  --run-config lr=0.001
```

### Benchmarking

```bash
# Compression benchmark
python -m fl.benchmarks.compression

# Privacy audit
python -m fl.privacy.membership_inference
```

### Deployment

```bash
# Deploy to Kubernetes
helm install fl-platform helm/fl-platform

# Scale clients
kubectl scale deployment fl-client --replicas=20
```

---

## Examples

### Custom Model

```python
import torch.nn as nn
from fl.models import SimpleCNN

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture
        
    def forward(self, x):
        # Your forward pass
        return x

# Use in training
model = CustomModel()
```

### Custom Aggregation

```python
from fl.robust_aggregation import RobustAggregator

class MyAggregator:
    def aggregate(self, client_updates):
        # Your aggregation logic
        return aggregated_update

aggregator = MyAggregator()
```

### Custom Attack

```python
from fl.attacks.byzantine import MaliciousClient

class MyAttack(MaliciousClient):
    def corrupt_gradients(self, state_dict, intensity):
        # Your attack logic
        return corrupted_dict

attacker = MyAttack(attack_type="custom")
```
