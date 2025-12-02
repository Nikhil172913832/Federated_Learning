# Architecture Documentation

## System Overview

This document provides a comprehensive breakdown of the federated learning platform architecture, component interactions, and design decisions.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Federated Learning Platform                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                    Control Plane                        │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │    │
│  │  │  SuperLink   │  │   MLflow     │  │  Dashboard  │ │    │
│  │  │ (Coordinator)│  │  (Tracking)  │  │ (Monitor)   │ │    │
│  │  └──────┬───────┘  └──────────────┘  └─────────────┘ │    │
│  └─────────┼────────────────────────────────────────────┘    │
│            │                                                     │
│  ┌─────────┴──────────────────────────────────────────────┐   │
│  │                    Data Plane                           │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────┐ │   │
│  │  │SuperNode1│  │SuperNode2│  │SuperNode3│  │  ...  │ │   │
│  │  │          │  │          │  │          │  │       │ │   │
│  │  │ClientApp │  │ClientApp │  │ClientApp │  │  N    │ │   │
│  │  │+ Data    │  │+ Data    │  │+ Data    │  │       │ │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └───────┘ │   │
│  └──────────────────────────────────────────────────────┘    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. SuperLink (Coordination Server)

**Purpose**: Central coordinator for federated learning rounds

**Responsibilities**:
- Client registration and management
- Round orchestration (start, monitor, complete)
- Message routing between server and clients
- Client selection for each round
- State management and checkpointing

**Technology Stack**:
- Flower Framework gRPC server
- Python asyncio for concurrent operations
- Protocol Buffers for serialization

**Key Interfaces**:
```python
# Client registration
register_client(client_id: str, capabilities: Dict) -> bool

# Round management
start_round(round_num: int, config: Dict) -> RoundID
get_round_status(round_id: RoundID) -> Status

# Message passing
send_parameters(client_ids: List[str], parameters: NDArray) -> None
receive_updates(round_id: RoundID) -> List[ClientUpdate]
```

### 2. SuperNode (Client Container)

**Purpose**: Isolated execution environment for client applications

**Features**:
- Docker containerization for isolation
- Resource management (CPU, memory, GPU)
- Local state persistence
- Secure communication with SuperLink

**Container Configuration**:
```yaml
services:
  supernode-1:
    image: fl-client:latest
    environment:
      - PARTITION_ID=0
      - NUM_PARTITIONS=3
      - SUPERLINK_ADDRESS=superlink:9093
    volumes:
      - ./client_stores/client-0:/app/storage
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
```

### 3. ClientApp (Federated Client)

**Purpose**: Execute local training on private data

**Workflow**:
```
1. Receive global model from SuperLink
2. Load local data partition
3. Train model for E local epochs
4. Compute model update/gradients
5. Apply privacy mechanisms (DP, masking)
6. Send update to SuperLink
7. Receive aggregated model
8. Repeat
```

**Code Structure**:
```python
@app.train()
def train(msg: Message, context: Context):
    # 1. Load model with global parameters
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    
    # 2. Load local data
    trainloader, _ = load_data(partition_id, num_partitions)
    
    # 3. Local training
    train_loss = train_fn(model, trainloader, epochs, lr, device)
    
    # 4. Apply privacy
    masked_params = mask_state_dict(model.state_dict(), enabled=True)
    
    # 5. Return update
    return Message(content={"arrays": masked_params, "metrics": metrics})
```

### 4. ServerApp (Aggregation Server)

**Purpose**: Aggregate client updates into global model

**Aggregation Strategies**:

#### FedAvg (Federated Averaging)
```python
def aggregate(updates: List[Tuple[NDArray, int]]) -> NDArray:
    """Weighted averaging based on dataset sizes"""
    total_samples = sum(num_samples for _, num_samples in updates)
    
    weighted_sum = None
    for params, num_samples in updates:
        weight = num_samples / total_samples
        if weighted_sum is None:
            weighted_sum = {k: v * weight for k, v in params.items()}
        else:
            for k in weighted_sum:
                weighted_sum[k] += params[k] * weight
    
    return weighted_sum
```

#### FedProx (Proximal Term)
```python
def aggregate_fedprox(updates, global_params, mu=0.01):
    """FedAvg with proximal regularization"""
    aggregated = fedavg_aggregate(updates)
    
    # Apply proximal term: moves less from global model
    for key in aggregated:
        aggregated[key] = (1 - mu) * aggregated[key] + mu * global_params[key]
    
    return aggregated
```

### 5. MLflow Server

**Purpose**: Experiment tracking and model versioning

**Tracked Artifacts**:
- Hyperparameters: learning rate, epochs, batch size
- Metrics: train loss, validation accuracy, communication cost
- Models: global model checkpoints per round
- Artifacts: confusion matrices, training curves

**Integration**:
```python
with start_run(experiment="fl", run_name=f"client-{partition_id}"):
    log_params({
        "partition_id": partition_id,
        "local_epochs": epochs,
        "lr": lr,
    })
    
    for round in range(num_rounds):
        # Training...
        log_metrics({
            "train_loss": loss,
            "accuracy": acc,
        }, step=round)
    
    log_model(model, "final_model")
```

### 6. Monitoring Dashboard

**Purpose**: Real-time visualization and control

**Features**:
- Live system metrics (CPU, memory, network)
- Training progress visualization
- Container health monitoring
- Configuration editor
- Start/stop controls

**Technology Stack**:
- Plotly Dash for UI
- WebSocket for real-time updates
- Docker SDK for container management

## Data Flow

### Training Round Flow

```
Round N:
1. ServerApp → SuperLink: "Start round N with config C"
2. SuperLink → SuperNodes: Select K clients, send global model
3. SuperNodes → ClientApps: Instantiate with model parameters
4. ClientApps: Load data, train locally for E epochs
5. ClientApps → SuperNodes: Return model updates + metrics
6. SuperNodes → SuperLink: Forward updates
7. SuperLink → ServerApp: Deliver all client updates
8. ServerApp: Aggregate updates → new global model
9. ServerApp → SuperLink: Store new model
10. Repeat from step 1 for round N+1
```

### Message Format

```python
class Message:
    """Flower message structure"""
    content: RecordDict  # Contains arrays and/or metrics
    metadata: Dict       # Round info, timestamps, etc.
    reply_to: Optional[Message]  # Request-response linking
    
class RecordDict:
    """Message payload"""
    arrays: Optional[ArrayRecord]      # Model parameters
    metrics: Optional[MetricRecord]    # Training metrics
    config: Optional[ConfigRecord]     # Configuration
```

## Security & Privacy

### 1. Differential Privacy (DP-SGD)

**Implementation**:
```python
from opacus import PrivacyEngine

def attach_dp_if_enabled(model, optimizer, dataloader, device, config):
    privacy_engine = PrivacyEngine()
    
    model, optimizer, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=config["noise_multiplier"],
        max_grad_norm=config["max_grad_norm"],
    )
    
    return privacy_engine, model, optimizer, dataloader
```

**Privacy Budget Tracking**:
- ε (epsilon): Privacy loss budget
- δ (delta): Probability of privacy breach
- Target: ε ≤ 3.0, δ ≤ 1e-5

### 2. Secure Aggregation

**Gradient Masking**:
```python
def mask_state_dict(state_dict, enabled=True):
    """Add random mask to gradients for secure aggregation"""
    if not enabled:
        return state_dict
    
    masked = {}
    for key, tensor in state_dict.items():
        mask = torch.randn_like(tensor) * 0.01
        masked[key] = tensor + mask
    
    return masked

def unmask_state_dict(state_dict, enabled=True):
    """Remove masks during aggregation (simplified)"""
    # In production: use cryptographic techniques
    return state_dict
```

### 3. TLS Encryption

**Certificate Generation**:
```bash
openssl req -x509 -newkey rsa:4096 \
    -keyout ca.key -out ca.crt \
    -days 365 -nodes \
    -subj "/CN=FL-CA"
```

**Client Configuration**:
```python
# With TLS
channel = grpc.secure_channel(
    'superlink:9093',
    grpc.ssl_channel_credentials(
        root_certificates=open('ca.crt', 'rb').read()
    )
)
```

## Data Partitioning

### IID Partitioning

```python
def iid_partition(dataset, num_clients):
    """Randomly distribute data evenly"""
    indices = np.random.permutation(len(dataset))
    partition_size = len(dataset) // num_clients
    
    partitions = []
    for i in range(num_clients):
        start = i * partition_size
        end = start + partition_size
        partitions.append(Subset(dataset, indices[start:end]))
    
    return partitions
```

### Non-IID Partitioning

#### Label Skew (Pathological)
```python
def label_skew_partition(dataset, num_clients, labels_per_client=2):
    """Each client gets only subset of classes"""
    label_indices = {i: [] for i in range(10)}  # MNIST: 10 classes
    
    for idx, (_, label) in enumerate(dataset):
        label_indices[label].append(idx)
    
    partitions = []
    for client in range(num_clients):
        # Randomly select labels_per_client classes
        selected_labels = np.random.choice(10, labels_per_client, replace=False)
        client_indices = []
        
        for label in selected_labels:
            client_indices.extend(label_indices[label])
        
        partitions.append(Subset(dataset, client_indices))
    
    return partitions
```

#### Dirichlet Partitioning
```python
def dirichlet_partition(dataset, num_clients, alpha=0.5):
    """Sample from Dirichlet distribution for realistic non-IID"""
    label_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_indices[label].append(idx)
    
    client_indices = [[] for _ in range(num_clients)]
    
    for label, indices in label_indices.items():
        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)
        
        # Distribute indices according to proportions
        np.random.shuffle(indices)
        for client in range(num_clients):
            start = proportions[client-1] if client > 0 else 0
            end = proportions[client]
            client_indices[client].extend(indices[start:end])
    
    return [Subset(dataset, indices) for indices in client_indices]
```

## Performance Optimization

### 1. Communication Efficiency

**Gradient Compression**:
```python
def compress_gradients(gradients, compression_ratio=0.1):
    """Top-k sparsification"""
    compressed = {}
    for key, tensor in gradients.items():
        flat = tensor.flatten()
        k = int(len(flat) * compression_ratio)
        
        # Keep only top-k largest absolute values
        topk_values, topk_indices = torch.topk(flat.abs(), k)
        
        compressed[key] = {
            'values': flat[topk_indices],
            'indices': topk_indices,
            'shape': tensor.shape
        }
    
    return compressed
```

### 2. Asynchronous Aggregation

```python
class AsyncAggregator:
    def __init__(self, staleness_threshold=3):
        self.global_model = None
        self.pending_updates = []
        self.current_round = 0
        self.staleness_threshold = staleness_threshold
    
    def receive_update(self, client_id, update, round_num):
        staleness = self.current_round - round_num
        
        if staleness <= self.staleness_threshold:
            # Apply staleness-aware weight
            weight = 1.0 / (1.0 + staleness)
            self.pending_updates.append((update, weight))
            
            # Aggregate when enough updates collected
            if len(self.pending_updates) >= self.min_updates:
                self.aggregate_and_update()
```

### 3. Client Selection

```python
class ClientSelector:
    def select_clients(self, available_clients, round_num):
        """Smart client selection based on multiple factors"""
        
        # Factor 1: Data size (importance sampling)
        data_sizes = [c.data_size for c in available_clients]
        probs = np.array(data_sizes) / sum(data_sizes)
        
        # Factor 2: Historical performance
        performance_scores = [c.avg_accuracy for c in available_clients]
        
        # Factor 3: Fairness (ensure all clients participate)
        participation_counts = [c.rounds_participated for c in available_clients]
        fairness_weights = 1.0 / (1.0 + np.array(participation_counts))
        
        # Combine factors
        final_scores = probs * 0.5 + performance_scores * 0.3 + fairness_weights * 0.2
        
        # Select top-k clients
        k = int(len(available_clients) * self.fraction)
        selected_idx = np.argsort(final_scores)[-k:]
        
        return [available_clients[i] for i in selected_idx]
```

## Scalability

### Horizontal Scaling

```yaml
# Scale to 100 clients
services:
  supernode:
    deploy:
      replicas: 100
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
```

### Vertical Scaling

```yaml
# More resources per client
services:
  supernode:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 16G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Monitoring & Observability

### Metrics Collected

**System Metrics**:
- CPU usage per container
- Memory consumption
- Network I/O (bytes sent/received)
- GPU utilization (if available)

**Training Metrics**:
- Per-round train loss
- Validation accuracy
- Convergence rate
- Communication cost (MB)

**Privacy Metrics**:
- Current privacy budget (ε, δ)
- Noise scale applied
- Gradient clipping frequency

### Logging Strategy

```python
import logging
import structlog

# Structured logging
logger = structlog.get_logger()

logger.info(
    "training_round_complete",
    round_num=5,
    train_loss=0.234,
    accuracy=0.891,
    clients_participated=8,
    aggregation_time_sec=2.3
)
```

## Failure Handling

### Client Dropout

```python
class RobustAggregator:
    def aggregate_with_dropout(self, updates, timeout=60):
        """Handle clients that don't respond in time"""
        
        # Wait for minimum number of clients
        min_clients = max(2, int(self.total_clients * 0.5))
        
        received = []
        deadline = time.time() + timeout
        
        while len(received) < min_clients and time.time() < deadline:
            update = self.receive_update(timeout=deadline - time.time())
            if update is not None:
                received.append(update)
        
        if len(received) >= min_clients:
            return self.aggregate(received)
        else:
            raise InsufficientClientsError(
                f"Only {len(received)} clients responded"
            )
```

### Checkpoint Recovery

```python
def resume_from_checkpoint(checkpoint_path):
    """Resume training from saved state"""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_round = checkpoint['round'] + 1
    
    logger.info(f"Resumed from round {start_round}")
    return start_round
```

## Design Decisions

### 1. Why Flower Framework?

**Pros**:
- Production-ready FL infrastructure
- Supports multiple aggregation strategies
- Excellent scalability
- Active community and development

**Cons**:
- Learning curve for custom strategies
- Some advanced features require workarounds

### 2. Why Docker Containerization?

**Pros**:
- Complete isolation between clients
- Easy deployment and scaling
- Reproducible environments
- Resource management

**Cons**:
- Overhead compared to processes
- More complex networking

### 3. Why MLflow for Tracking?

**Pros**:
- Industry standard for ML experiments
- Model registry and versioning
- Easy integration
- Great UI for visualization

**Alternatives Considered**:
- Weights & Biases: More features but requires account
- TensorBoard: Simpler but less comprehensive
- Custom solution: Too much development effort

## Future Enhancements

1. **Federated Analytics**: Privacy-preserving data analysis
2. **Vertical Federated Learning**: Different features per client
3. **Split Learning**: Split model between client and server
4. **Blockchain Integration**: Decentralized coordination
5. **Edge Device Support**: Mobile and IoT deployment

## References

- [Flower Framework Documentation](https://flower.ai/docs/)
- [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- [Federated Optimization: Distributed Machine Learning for On-Device Intelligence](https://arxiv.org/abs/1610.02527)
