# Personalized Federated Learning with Knowledge Distillation

## Overview

Production-ready personalized FL system with knowledge distillation, enabling model-agnostic per-client personalization through adapter-based training.

## Key Features

### Core Capabilities

- **Knowledge Distillation**: Train personalized student models using global teacher guidance
- **Adapter-Based Personalization**: LoRA, bottleneck adapters, prefix tuning
- **Model Agnostic**: Each client can have different architectures
- **Communication Efficient**: Only share global parameters
- **Privacy Preserving**: Differential privacy and secure aggregation support
- **Byzantine Robust**: Anomaly detection for malicious update filtering

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FL Server (Orchestrator)                  │
│  - Global model management                                   │
│  - Client selection & scheduling                             │
│  - Aggregation (FedAvg/FedProx/FedAdam)                     │
│  - Anomaly detection                                         │
│  - Checkpointing & versioning                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ├─────────────┬─────────────┐
                            ▼             ▼             ▼
            ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
            │    Client 1       │ │    Client 2       │ │    Client N       │
            │                   │ │                   │ │                   │
            │ Global Model      │ │ Global Model      │ │ Global Model      │
            │      +            │ │      +            │ │      +            │
            │ LoRA Adapters     │ │ Bottleneck        │ │ LoRA Adapters     │
            │ (personalized)    │ │ Adapters          │ │ (personalized)    │
            │                   │ │ (personalized)    │ │                   │
            │ KD Training       │ │ KD Training       │ │ KD Training       │
            └───────────────────┘ └───────────────────┘ └───────────────────┘
                    │                     │                     │
                    │                     │                     │
            Local Storage         Local Storage         Local Storage
            (Personalized         (Personalized         (Personalized
             Params)               Params)               Params)
```

## Components

### Knowledge Distillation (`distillation.py`)

Implements various distillation strategies:

#### DistillationConfig
```python
@dataclass
class DistillationConfig:
    temperature: float = 2.0          # Softmax temperature (T ∈ [1, 5])
    lambda_kd: float = 0.5            # KD loss weight (λ ∈ [0.2, 0.7])
    lambda_ce: float = 0.5            # CE loss weight
    kd_loss_type: str = "kl"          # "kl", "mse", or "cosine"
    use_hard_labels: bool = True      # Include supervised CE loss
    feature_distillation: bool = False # Feature-level distillation
    adaptive_temperature: bool = False # Adapt T based on confidence
```

#### Loss Function

The combined loss is:

```
L = (1-λ) * L_CE + λ * T² * L_KD + μ * L_reg + β * L_feat

Where:
- L_CE: Cross-entropy with hard labels
- L_KD: KL divergence between soft targets
- L_reg: Optional regularization (prevents catastrophic personalization)
- L_feat: Optional feature-level distillation
- T²: Scaling factor for gradient consistency
```

#### LocalDistillationTrainer

Client-side training with distillation:

```python
trainer = LocalDistillationTrainer(
    student_model=personalized_model,
    teacher_model=global_model,
    distillation_config=config,
    optimizer=optimizer,
    device=device
)

metrics = trainer.train_epoch(dataloader)
```

### 2. Adapters (`adapters.py`)

Parameter-efficient fine-tuning (PEFT) modules:

#### LoRA (Low-Rank Adaptation)

```python
# Add LoRA to a linear layer
lora_linear = LoRALinear(
    linear=nn.Linear(768, 768),
    rank=4,           # Decomposition rank
    alpha=16.0,       # Scaling factor
    dropout=0.0,
    freeze_base=True  # Freeze original weights
)

# Forward: h = Wx + (BA)x * (alpha/rank)
output = lora_linear(input)
```

**Parameters**: Original model has `d_in * d_out` params. LoRA adds only `rank * (d_in + d_out)`.

For `rank=4`, `d=768`: Original = 589,824 params, LoRA = 6,144 params (1% overhead!)

#### Bottleneck Adapter

```python
adapter = BottleneckAdapter(
    input_dim=768,
    bottleneck_dim=64,  # Hidden dimension
    activation="relu",
    dropout=0.1
)

# Forward: x + down_proj(activation(up_proj(x)))
output = adapter(input)
```

#### ModelWithAdapters

Automatically add adapters to target modules:

```python
adapted_model = ModelWithAdapters(
    base_model=pretrained_model,
    adapter_config={
        "type": "lora",
        "rank": 4,
        "alpha": 16.0,
    },
    target_modules=["attention.q", "attention.k", "attention.v"]
)

# Freeze base model
adapted_model.freeze_base_model()

# Get/load only adapter params
adapter_params = adapted_model.get_adapter_params()
adapted_model.load_adapter_params(adapter_params)
```

### 3. Personalized Client (`personalized_client.py`)

Client-side FL training:

```python
config = PersonalizedClientConfig(
    client_id="client_1",
    local_epochs=1,
    learning_rate=1e-3,
    adapter_type="lora",
    adapter_rank=4,
    clip_norm=1.0,
    min_samples_for_personalization=50,
    storage_path=Path("./storage/client_1")
)

client = PersonalizedFLClient(config, base_model, device)

# Receive global model
client.receive_global_model(global_weights, round_idx=0)

# Local training with KD
metrics = client.local_train(train_loader, val_loader)

# Get update (only shared params)
update = client.get_model_update()
```

**Features**:
- Automatic adapter initialization and loading
- Knowledge distillation from global teacher
- Gradient clipping and compression
- Personalized param storage (SQLite)
- Regularization to prevent overfitting

### 4. Personalized Server (`personalized_server.py`)

Server-side orchestration:

```python
config = PersonalizedServerConfig(
    total_rounds=1000,
    clients_per_round=50,
    aggregation_strategy="fedavg",  # "fedavg", "fedadam", "fedprox"
    weighted_aggregation=True,
    enable_anomaly_detection=True,
    checkpoint_every_n_rounds=10
)

server = PersonalizedFLServer(config, global_model, device)

# Register clients
server.register_client("client_1", {"dataset_size": 1000})

# Select clients for round
selected = server.select_clients(round_idx=0)

# Get global weights
global_weights = server.get_global_model_weights()

# Aggregate updates
client_updates = {...}  # Dict[client_id, update]
client_weights = {...}  # Dict[client_id, weight]
stats = server.aggregate_updates(client_updates, client_weights, round_idx=0)

# Evaluate
metrics = server.evaluate_global_model(test_loader)

# Checkpoint
server.save_checkpoint(round_idx=0)
```

**Aggregation Strategies**:

- **FedAvg**: Weighted average of client updates
  ```
  θ_global = Σ (w_i * θ_i) / Σ w_i
  ```

- **FedAdam**: Server-side adaptive optimization
  ```
  m_t = β₁ * m_{t-1} + (1-β₁) * Δθ
  v_t = β₂ * v_{t-1} + (1-β₂) * Δθ²
  θ_global = θ_global - η * m̂_t / (√v̂_t + ε)
  ```

**Anomaly Detection**:
- Compute L2 norm of each update
- Reject updates with norm > median * threshold (default 3.0)
- Adaptive thresholding based on history

### 5. Orchestrator (`orchestrator.py`)

High-level API for running experiments:

```python
# Create config
config = create_personalized_fl_config(
    total_rounds=100,
    clients_per_round=10,
    local_epochs=1,
    learning_rate=1e-3,
    adapter_type="lora",
    adapter_rank=4,
    temperature=2.0,
    lambda_kd=0.5
)

# Create orchestrator
orchestrator = PersonalizedFLOrchestrator(
    config=config,
    model_fn=lambda: YourModel(),
    client_data_loaders=client_loaders,
    global_test_loader=test_loader
)

# Run training
results = orchestrator.run()
```

**Features**:
- Automatic client initialization
- Progressive distillation (curriculum learning)
- Comprehensive metrics tracking
- Personalization gain computation
- Result export (JSON, checkpoints)

## Usage Guide

### Quick Start

```python
from fl.personalization import (
    create_personalized_fl_config,
    PersonalizedFLOrchestrator
)

# 1. Prepare data loaders
client_loaders = {
    "client_1": {"train": loader1_train, "val": loader1_val},
    "client_2": {"train": loader2_train, "val": loader2_val},
    # ...
}

# 2. Create config
config = create_personalized_fl_config(
    total_rounds=100,
    clients_per_round=10,
    adapter_type="lora",
    adapter_rank=4
)

# 3. Run experiment
orchestrator = PersonalizedFLOrchestrator(
    config=config,
    model_fn=lambda: YourModel(),
    client_data_loaders=client_loaders
)

results = orchestrator.run()
```

### Advanced Configuration

#### Custom Distillation

```python
from fl.personalization import DistillationConfig

distillation_config = DistillationConfig(
    temperature=3.0,
    lambda_kd=0.6,
    kd_loss_type="kl",
    use_hard_labels=True,
    feature_distillation=True,
    feature_lambda=0.1,
    adaptive_temperature=True
)
```

#### Custom Aggregation

```python
from fl.personalization import PersonalizedServerConfig

server_config = PersonalizedServerConfig(
    aggregation_strategy="fedadam",
    server_learning_rate=0.01,
    server_adam_beta1=0.9,
    server_adam_beta2=0.999,
    enable_anomaly_detection=True,
    anomaly_threshold_multiplier=3.0
)
```

#### Privacy & Security

```python
server_config = PersonalizedServerConfig(
    enable_differential_privacy=True,
    dp_noise_multiplier=0.1,
    dp_clip_norm=1.0,
    enable_secure_aggregation=True
)
```

## Hyperparameter Guidelines

### Starting Configuration

```python
{
    # FL settings
    "total_rounds": 100-200,
    "clients_per_round": 50-200,
    "local_epochs": 1-5,
    
    # Learning rates
    "client_lr": 1e-3 (adapters), 1e-4 (shared head),
    "server_lr": 1.0 (FedAvg), 0.001 (FedAdam),
    
    # Adapters
    "adapter_type": "lora",
    "lora_rank": 4-8,
    "lora_alpha": 16.0,
    "bottleneck_dim": 64-256,
    
    # Distillation
    "temperature": 2.0-3.0,
    "lambda_kd": 0.3-0.6,
    
    # Robustness
    "clip_norm": 1.0,
    "weight_decay": 1e-5 to 1e-4,
    
    # Personalization
    "min_samples": 50-200,
    "warmup_rounds": 5-10,
}
```

### Tuning Guidelines

| Problem | Solution |
|---------|----------|
| Global loss increases | Check aggregation, reduce local epochs, enable clipping |
| Personalization doesn't help | Lower λ_kd, increase T, add regularization |
| Adapters not learning | Increase adapter rank, raise learning rate |
| Updates diverge | Enable clipping, reduce local epochs, increase client samples |
| DP hurts performance | Reduce noise, increase clients/round, relax privacy budget |

## Evaluation Metrics

### Computed Metrics

```python
{
    # Client metrics (per-client)
    "train_loss": ...,
    "val_accuracy": ...,
    "update_norm": ...,
    "kd_loss": ...,
    "ce_loss": ...,
    
    # Server metrics (aggregated)
    "mean_update_norm": ...,
    "std_update_norm": ...,
    "mean_update_similarity": ...,  # Divergence measure
    "num_clients": ...,
    
    # Global evaluation
    "global_accuracy": ...,
    "global_loss": ...,
    
    # Personalization gains
    "mean_gain": ...,  # Δ = personal_acc - global_acc
    "median_gain": ...,
    "max_gain": ...,
    "mean_personal_acc": ...,
    "mean_global_acc": ...,
}
```

### Key Metrics to Monitor

1. **Personalization Gain**: `personal_acc - global_acc`
   - Should be positive (personalization helps)
   - Target: +5-15% improvement

2. **Update Divergence**: Cosine similarity between client updates
   - Too high = clients not learning differently
   - Too low = catastrophic divergence

3. **Teacher-Student Agreement**: Prediction agreement rate
   - Target: 70-90% agreement
   - Lower = student learning something different

4. **Fairness**: Worst-10% client performance
   - Monitor tail performance
   - Ensure all clients benefit

## Implementation Details

### Communication Protocol

```
Round 0:
  Server → Clients: global_weights, config (T, λ)
  Clients → Server: Nothing (initial round)

Round 1..N:
  1. Server selects clients
  2. Server → Clients: global_weights_v{n}
  3. Clients:
     - Load global weights into shared params
     - Load local personalized params from storage
     - Train with KD (local epochs)
     - Extract shared param updates only
     - Clip and compress updates
  4. Clients → Server: {updates, metadata}
  5. Server:
     - Filter anomalous updates
     - Aggregate with weighted average
     - Update global model
     - Checkpoint and evaluate
```

### Storage Structure

```
outputs/
├── experiment_name/
│   ├── checkpoints/
│   │   ├── round_0.pt
│   │   ├── round_10.pt
│   │   └── ...
│   ├── personalization/
│   │   ├── client_1/
│   │   │   └── personalization.db (SQLite)
│   │   ├── client_2/
│   │   │   └── personalization.db
│   │   └── ...
│   ├── metrics.json
│   ├── server_metrics.json
│   ├── results.json
│   └── training.log
```

### Checkpoint Format

```python
{
    "round": 10,
    "model_version": 10,
    "model_state_dict": {...},
    "server_optimizer_state": {...},
    "config": {...},
    "global_metrics_history": [...],
    "timestamp": 1234567890.0
}
```

## Best Practices

### 1. Data Preparation

- ✅ Ensure minimum samples per client (50-200)
- ✅ Create heterogeneous splits (Dirichlet, label skew)
- ✅ Reserve validation/test sets per client
- ✅ Use consistent preprocessing/augmentation

### 2. Model Architecture

- ✅ Start with proven architectures (ResNet, ViT, BERT)
- ✅ Add adapters to attention layers and final layers
- ✅ Keep backbone frozen initially
- ✅ Use layer normalization before adapters

### 3. Training

- ✅ Start simple: FedAvg + local KD (no adapters)
- ✅ Add adapters once baseline works
- ✅ Use progressive distillation (curriculum learning)
- ✅ Enable gradient clipping (critical for stability)
- ✅ Monitor update norms and divergence

### 4. Debugging

- ✅ Test single client first
- ✅ Verify serialization/deserialization
- ✅ Check adapter params are not uploaded
- ✅ Validate global model updates separately
- ✅ Use deterministic seeds for reproducibility

### 5. Production

- ✅ Enable anomaly detection
- ✅ Implement client timeouts and retries
- ✅ Version models and configs
- ✅ Log extensively
- ✅ Checkpoint frequently
- ✅ Monitor resource usage (compute, bandwidth)

## Troubleshooting

### Common Issues

**Q: Global loss increases after aggregation**

A: Likely causes:
1. Unbounded client updates → Enable clipping
2. Anomalous clients → Enable anomaly detection
3. Too many local epochs → Reduce to 1-3
4. Wrong aggregation weights → Check weighted_aggregation

**Q: Personalization doesn't improve accuracy**

A: Likely causes:
1. Too much distillation (λ_kd too high) → Lower to 0.3-0.5
2. Insufficient local data → Increase min_samples
3. Adapters too small → Increase rank/bottleneck
4. Frozen backbone preventing learning → Selectively unfreeze

**Q: Adapters not training**

A: Likely causes:
1. Learning rate too low → Increase to 1e-3
2. Upstream layers frozen → Check freeze_backbone
3. Adapter rank too small → Increase to 8-16
4. No gradient flow → Verify backward pass

**Q: Training diverges**

A: Likely causes:
1. No gradient clipping → Enable clip_norm=1.0
2. Local learning rate too high → Reduce by 10x
3. Too many local epochs → Set to 1
4. Heterogeneity too extreme → Reduce alpha in Dirichlet

## Examples

See `examples/personalized_fl_example.py` for a complete working example with CIFAR-10.

## References

### Papers

1. **FedAvg**: McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data" (2017)
2. **FedProx**: Li et al. "Federated Optimization in Heterogeneous Networks" (2020)
3. **FedAdam**: Reddi et al. "Adaptive Federated Optimization" (2021)
4. **Knowledge Distillation**: Hinton et al. "Distilling the Knowledge in a Neural Network" (2015)
5. **FedDF**: Lin et al. "Ensemble Distillation for Robust Model Fusion in Federated Learning" (2020)
6. **LoRA**: Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
7. **Adapters**: Houlsby et al. "Parameter-Efficient Transfer Learning for NLP" (2019)

### Resources

- [Flower Framework](https://flower.dev/)
- [PEFT Library](https://github.com/huggingface/peft)
- [FL Benchmarks](https://github.com/FedML-AI/FedML)

## License

This implementation is part of the Federated Learning platform and follows the same license.
