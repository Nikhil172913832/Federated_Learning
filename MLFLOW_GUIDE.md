# 📊 MLflow Integration Guide

## Overview

This Federated Learning platform automatically logs all training metrics and parameters to MLflow, giving you complete visibility into your experiments.

## Accessing MLflow

1. **Ensure the platform is running**:
   ```bash
   docker ps | grep mlflow
   # Should show: complete-mlflow-1   Up X minutes   0.0.0.0:5000->5000/tcp
   ```

2. **Open MLflow UI**:
   - Navigate to: http://localhost:5000
   - No authentication required for local deployment

## MLflow Interface Walkthrough

### 1. Experiments Page

When you first open MLflow, you'll see the **Experiments** list:

- **Default Experiment**: Auto-created by MLflow
- **fl Experiment**: Our Federated Learning experiment
  - Click on "fl" to view all runs

### 2. Runs Overview

Inside the "fl" experiment, you'll see multiple runs:

```
Run Name        | Status    | Start Time          | Parameters
----------------|-----------|---------------------|------------------
server          | FINISHED  | 2025-10-05 14:30:00 | num_rounds=3, lr=0.01
client-0        | FINISHED  | 2025-10-05 14:30:05 | partition_id=0, lr=0.01
client-1        | FINISHED  | 2025-10-05 14:30:05 | partition_id=1, lr=0.01
client-2        | FINISHED  | 2025-10-05 14:30:05 | partition_id=2, lr=0.01
```

### 3. Server Run Details

Click on the **"server"** run to see:

**Parameters**:
- `num_rounds`: 3
- `fraction_train`: 0.5
- `lr`: 0.01
- `num_partitions`: 3

**Metrics**:
- `training_complete`: 1 (at step 3)

**Metadata**:
- Start time
- End time
- Duration
- Status (FINISHED/FAILED/RUNNING)

### 4. Client Run Details

Click on any **"client-X"** run (e.g., "client-0") to see:

**Parameters**:
- `partition_id`: 0 (which data partition this client used)
- `num_partitions`: 3
- `local_epochs`: 1
- `lr`: 0.01

**Metrics** (tracked across rounds):
| Step (Round) | train_loss |
|--------------|------------|
| 0            | 2.45       |
| 1            | 1.82       |
| 2            | 1.34       |

**Chart**:
- X-axis: Step (round number)
- Y-axis: train_loss
- Shows convergence over time

## Viewing Training Metrics

### Method 1: Individual Run View

1. Click on a client run (e.g., "client-0")
2. Scroll to the "Metrics" section
3. Click on "train_loss"
4. See the chart showing loss over rounds

### Method 2: Compare All Clients

1. In the runs list, check the boxes for client-0, client-1, and client-2
2. Click the **"Compare"** button
3. You'll see:
   - **Parallel Coordinates Plot**: Visual comparison of parameters
   - **Scatter Plot**: Metrics comparison
   - **Box Plot**: Distribution of metrics across clients
   - **Line Plot**: Loss curves for all clients overlaid

### Method 3: Download Data

1. Select the runs you want to export
2. Click **"Compare"** → **"Download CSV"**
3. Open in Excel/Google Sheets for custom analysis

## Understanding the Metrics

### Train Loss

**What it means**:
- Lower is better
- Measures how well the model fits the training data
- Should decrease over rounds (convergence)

**Expected pattern**:
```
Round 0: ~2.5 (random initialization)
Round 1: ~1.8 (learning)
Round 2: ~1.3 (converging)
Round 3: ~1.1 (near optimal)
```

**Interpretations**:
- ✅ **Decreasing loss**: Model is learning correctly
- ⚠️ **Fluctuating loss**: May need lower learning rate
- ❌ **Increasing loss**: Learning rate too high or data issue
- ⚠️ **Plateau early**: May need more rounds or higher learning rate

### Loss Differences Across Clients

It's normal for clients to have different losses:

- **Client 0 loss: 1.2** ✅
- **Client 1 loss: 1.5** ✅
- **Client 2 loss: 1.3** ✅

This happens because:
1. Each client has different data
2. Data may have different difficulty levels
3. Random initialization differences

**When to worry**:
- One client's loss >> others (data quality issue)
- All clients have high loss (model architecture or hyperparameter issue)

## Advanced MLflow Features

### 1. Search and Filter Runs

Use the search bar to filter runs:

```
# Find runs with specific parameter
params.partition_id = "0"

# Find successful runs only
attributes.status = "FINISHED"

# Find runs with low loss
metrics.train_loss < 1.5

# Combine conditions
params.lr = "0.01" AND metrics.train_loss < 2.0
```

### 2. Add Tags

Add custom tags to runs for organization:

1. Click on a run
2. Click "Add Tag"
3. Examples:
   - `experiment_type: baseline`
   - `data_distribution: iid`
   - `notes: first_successful_run`

### 3. Add Notes

Document your experiments:

1. Click on a run
2. Find the "Notes" section
3. Write observations:
   ```
   This run used IID data distribution with 3 clients.
   Performance was good with steady convergence.
   Next: Try non-IID data.
   ```

### 4. Register Models

Although we save models as files, you can track them in MLflow:

1. Go to server run
2. Click "Register Model" (if artifacts are logged)
3. Name it (e.g., "PneumoniaDetector")
4. Track versions over experiments

## Integration with Training Code

### Server-Side Logging

Location: `complete/fl/fl/server_app.py`

```python
from fl.tracking import start_run, log_params, log_metrics

# Start MLflow run
with start_run(experiment="fl", run_name="server"):
    # Log hyperparameters
    log_params({
        "num_rounds": num_rounds,
        "fraction_train": fraction_train,
        "lr": lr,
        "num_partitions": 3,
    })
    
    # ... training code ...
    
    # Log completion metric
    log_metrics({"training_complete": 1}, step=num_rounds)
```

### Client-Side Logging

Location: `complete/fl/fl/client_app.py`

```python
from fl.tracking import start_run, log_params, log_metrics

# Start MLflow run for this client
with start_run(experiment="fl", run_name=f"client-{partition_id}"):
    # Log client configuration
    log_params({
        "partition_id": partition_id,
        "num_partitions": num_partitions,
        "local_epochs": int(run_cfg["local-epochs"]),
        "lr": msg.content["config"]["lr"],
    })
    
    # Train and get loss
    train_loss = train_fn(...)
    
    # Log training loss for this round
    round_idx = int(context.run_config.get("round", 0))
    log_metrics({"train_loss": train_loss}, step=round_idx)
```

## Automatic Tracking

The platform automatically tracks:

✅ **Experiment Name**: "fl"  
✅ **Run Names**: "server", "client-0", "client-1", "client-2"  
✅ **Parameters**: All hyperparameters  
✅ **Metrics**: Training loss per round  
✅ **Timestamps**: Start/end times  
✅ **Status**: Running/Finished/Failed  

## Troubleshooting MLflow

### "No Experiment Found"

**Problem**: MLflow UI shows only "Default" experiment

**Solution**:
```bash
# Check if training actually ran
docker logs complete-superexec-serverapp-1 | grep -i mlflow

# Ensure MLFLOW_TRACKING_URI is set
docker exec complete-superexec-serverapp-1 env | grep MLFLOW
# Should show: MLFLOW_TRACKING_URI=http://mlflow:5000
```

### "Connection Refused"

**Problem**: Can't access http://localhost:5000

**Solutions**:
```bash
# Check if MLflow container is running
docker ps | grep mlflow

# Check MLflow logs
docker logs complete-mlflow-1

# Restart MLflow
docker restart complete-mlflow-1

# Check port binding
netstat -tuln | grep 5000
```

### Metrics Not Showing

**Problem**: Run exists but no metrics

**Possible causes**:
1. Training didn't complete successfully
2. MLflow logging failed silently
3. Network issue between containers

**Debug steps**:
```bash
# Check client logs for MLflow errors
docker logs complete-superexec-clientapp-1-1 | grep -i "mlflow\|error"

# Check if mlflow package is installed
docker exec complete-superexec-clientapp-1-1 pip list | grep mlflow

# Test MLflow connection from inside container
docker exec complete-superexec-clientapp-1-1 python -c "
import os
print(os.environ.get('MLFLOW_TRACKING_URI'))
"
```

### Duplicate Runs

**Problem**: Multiple runs with same name

**Explanation**: This is normal! Each time you run training, new runs are created.

**Best practice**:
- Use tags to identify specific experiments
- Add notes to document each run
- Use search/filter to find relevant runs

## Exporting Data

### Export All Metrics

```python
import mlflow
import pandas as pd

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Get experiment
experiment = mlflow.get_experiment_by_name("fl")

# Get all runs
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Save to CSV
runs.to_csv("all_runs.csv", index=False)

# Extract specific metrics
client_runs = runs[runs['tags.mlflow.runName'].str.contains('client')]
print(client_runs[['tags.mlflow.runName', 'metrics.train_loss']])
```

### Export Model

```bash
# Copy final model from server container
docker cp complete-superexec-serverapp-1:/app/final_model.pt ./models/run_$(date +%Y%m%d_%H%M%S).pt

# Load in Python
import torch
model_state = torch.load('final_model.pt')
```

## Best Practices

### 1. Organize Experiments

Create separate experiments for different scenarios:
- `fl_baseline` - IID data, standard settings
- `fl_noniid` - Non-IID data experiments
- `fl_privacy` - Differential privacy experiments

### 2. Use Consistent Naming

- Server run: Always "server"
- Client runs: "client-{partition_id}"
- Add meaningful tags for variants

### 3. Document Everything

- Add notes to each run explaining:
  - What you changed
  - Why you changed it
  - What you expected
  - What actually happened

### 4. Compare Systematically

- Always compare against a baseline
- Change one variable at a time
- Run multiple times to check reproducibility

### 5. Archive Important Runs

- Add tag: `status: production`
- Add tag: `best_model: true`
- Export data before cleanup

## Summary

With MLflow integration, you get:

✅ **Automatic tracking** of all experiments  
✅ **Visual comparison** of different runs  
✅ **Reproducibility** through parameter logging  
✅ **Easy sharing** via exports and UI  
✅ **Historical analysis** of all past experiments  

Your federated learning experiments are now fully tracked and analyzable! 🎉

---

**Next**: Run training and explore your metrics at http://localhost:5000
