# MLflow Fresh Logs Guide

This guide explains how to get fresh MLflow logs for each training run.

## Problem
MLflow uses a persistent Docker volume (`mlflow-data`) that stores all experiment data across restarts, causing old logs to accumulate.

## Solutions

### Option 1: Timestamped Experiments (✅ Already Applied)

The tracking code has been updated to automatically create unique experiment names with timestamps for each run.

**How it works:**
- Each training run creates a new experiment like `fl_20251006_143025`
- Old experiments remain visible but new data goes to a fresh experiment
- No manual cleanup needed

**Usage:**
```bash
cd complete
flwr run fl local-deployment --stream
```

Each run will appear as a separate experiment in MLflow UI.

---

### Option 2: Complete Fresh Start Script

Use the `fresh-training.sh` script to clean everything and start fresh training.

**Usage:**
```bash
cd complete
./fresh-training.sh
```

**What it does:**
1. Stops all containers
2. Removes MLflow volume (deletes all old logs)
3. Starts containers fresh
4. Automatically runs training

---

### Option 3: Clean MLflow Only

Use `clean-mlflow.sh` to clean only MLflow data without stopping other containers.

**Usage:**
```bash
cd complete
./clean-mlflow.sh
```

**What it does:**
1. Stops MLflow container
2. Removes MLflow volume
3. Restarts MLflow container
4. Keeps all other services running

Then run training manually:
```bash
flwr run fl local-deployment --stream
```

---

### Option 4: Manual Cleanup

If you prefer manual control:

```bash
# Stop all services
docker compose -f compose-with-ui.yml down

# Remove MLflow volume
docker volume rm complete_mlflow-data

# Start services
docker compose -f compose-with-ui.yml up -d

# Run training
flwr run fl local-deployment --stream
```

---

## Comparison

| Method | Pros | Cons |
|--------|------|------|
| **Timestamped Experiments** | Automatic, keeps history, no manual steps | Accumulates experiments over time |
| **Fresh Start Script** | Fully automated, completely clean | Deletes all history |
| **Clean MLflow Only** | Fast, keeps other services running | Requires manual training start |
| **Manual Cleanup** | Full control | Most steps to remember |

## Recommendation

**Use Option 1 (Timestamped Experiments)** for day-to-day development - it's automatic and keeps your experiment history organized.

**Use Option 2 (Fresh Start Script)** when you want to completely reset and don't need old logs.

## Accessing MLflow UI

After starting services, access MLflow at:
- **MLflow UI:** http://localhost:5000
- **Platform UI:** http://localhost:8050

## Viewing Experiments

In MLflow UI:
1. Click on "Experiments" in the left sidebar
2. Each timestamped experiment (e.g., `fl_20251006_143025`) represents one training run
3. Click on an experiment to see its runs and metrics
4. Compare multiple experiments using the "Compare" feature
