# 🏗️ Platform Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   FEDERATED LEARNING PLATFORM ARCHITECTURE                      │
│                        (All components in Docker containers)                     │
└─────────────────────────────────────────────────────────────────────────────────┘

                                  ┌──────────────┐
                                  │  Your Browser│
                                  └──────┬───────┘
                                         │
                   ┌─────────────────────┼─────────────────────┐
                   │                     │                     │
                   ▼                     ▼                     ▼
           ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
           │ Platform UI  │     │   MLflow UI  │     │  SuperLink   │
           │ Port 8050    │     │  Port 5000   │     │  API:9093    │
           └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
                  │                    │                     │
                  │ Monitors           │ Tracks              │ Coordinates
                  ▼                    ▼                     ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                        FEDERATED LEARNING SYSTEM                             ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

                            ┌─────────────────┐
                            │   SuperLink     │
                            │  Coordination   │
                            │   Port 9093     │
                            └────────┬────────┘
                                     │
                ┏━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━┓
                ▼                                         ▼
        ┌───────────────┐                         ┌──────────────┐
        │  SuperExec    │                         │  SuperNodes  │
        │  ServerApp    │◄────Aggregation────────►│   1, 2, 3    │
        │  (Container)  │                         │ (Containers) │
        └───────┬───────┘                         └──────┬───────┘
                │                                        │
                │ Runs Server                            │ Manage Clients
                │ FL Logic                               │
                ▼                                        ▼
        ┌───────────────┐                   ┌────────────────────────┐
        │ server_app.py │                   │  SuperExec ClientApps  │
        │               │                   │    1, 2, 3             │
        │ • FedAvg      │                   │   (Containers)         │
        │ • Aggregation │                   └───────────┬────────────┘
        │ • MLflow Log  │                               │
        └───────┬───────┘                               │ Run Client FL Logic
                │                                        │
                │                                        ▼
                │                               ┌────────────────────┐
                │                               │  client_app.py     │
                │                               │  (3 instances)     │
                │                               │                    │
                │                               │  • Local Training  │
                │                               │  • MLflow Logging  │
                │                               │  • Send Updates    │
                │                               └─────────┬──────────┘
                │                                         │
                │                                         │
                │                                         ▼
                │                               ┌─────────────────────┐
                │                               │   Data Partitions   │
                │                               │                     │
                │                               │  Client 0: Part 0/3 │
                │                               │  Client 1: Part 1/3 │
                │                               │  Client 2: Part 2/3 │
                │                               │                     │
                │                               │ MedMNIST (Pneumonia)│
                │                               └─────────────────────┘
                │
                ▼
        ┌──────────────────┐
        │  final_model.pt  │
        │  (Trained Model) │
        └──────────────────┘

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                           MONITORING & TRACKING                              ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

    ┌─────────────────────┐              ┌─────────────────────────────┐
    │   Platform UI       │              │        MLflow               │
    │   (Port 8050)       │              │      (Port 5000)            │
    ├─────────────────────┤              ├─────────────────────────────┤
    │                     │              │                             │
    │ • Container Status  │              │ Experiment: "fl"            │
    │ • CPU Usage         │              │                             │
    │ • Memory Usage      │              │ Runs:                       │
    │ • Network I/O       │              │  • server                   │
    │ • Disk I/O          │              │  • client-0                 │
    │ • Uptime            │              │  • client-1                 │
    │ • Health Checks     │              │  • client-2                 │
    │                     │              │                             │
    │ Real-time Updates   │              │ Metrics:                    │
    │ Every 5 seconds     │              │  • train_loss (per round)   │
    │                     │              │  • training_complete        │
    │                     │              │                             │
    └─────────────────────┘              │ Parameters:                 │
                                         │  • lr, num_rounds           │
                                         │  • partition_id             │
                                         │  • local_epochs             │
                                         │                             │
                                         │ Charts & Comparisons        │
                                         │ Export to CSV               │
                                         └─────────────────────────────┘

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                          FEDERATED LEARNING FLOW                             ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

    Round 0:
    ────────
    Server: Create global model → Send to selected clients (50%)
    Client 0: Receive model → Train on partition 0 → Send updates
    Client 1: Receive model → Train on partition 1 → Send updates
    Server: Aggregate updates (FedAvg) → Update global model
    MLflow: Log train_loss for each client at step=0
    
    Round 1:
    ────────
    Server: Send updated global model → Selected clients
    Client 0: Receive model → Train → Send updates
    Client 2: Receive model → Train → Send updates
    Server: Aggregate → Update global model
    MLflow: Log train_loss for each client at step=1
    
    Round 2:
    ────────
    Server: Send updated global model → Selected clients
    Client 1: Receive model → Train → Send updates
    Client 2: Receive model → Train → Send updates
    Server: Aggregate → Final global model
    MLflow: Log train_loss for each client at step=2
    
    Complete:
    ─────────
    Server: Save final_model.pt
    MLflow: Log training_complete=1

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                              DOCKER CONTAINERS                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

    Container Name                      | Image             | Ports      | Purpose
    ────────────────────────────────────┼───────────────────┼────────────┼─────────────────
    complete-superlink-1                | flwr/superlink    | 9093       | FL Coordination
    complete-supernode-1-1              | flwr/supernode    | 9094       | Client Manager 1
    complete-supernode-2-1              | flwr/supernode    | 9095       | Client Manager 2
    complete-supernode-3-1              | flwr/supernode    | 9096       | Client Manager 3
    complete-superexec-serverapp-1      | Custom Build      | -          | FL Server Logic
    complete-superexec-clientapp-1-1    | Custom Build      | -          | FL Client 1
    complete-superexec-clientapp-2-1    | Custom Build      | -          | FL Client 2
    complete-superexec-clientapp-3-1    | Custom Build      | -          | FL Client 3
    complete-fl-platform-ui-1           | Custom Build      | 8050       | Monitoring UI
    complete-mlflow-1                   | Custom Build      | 5000       | Experiment Track

    Total: 10 containers

    Volumes:
    ────────
    superlink-state → Persistent state for SuperLink
    mlflow-data     → MLflow experiments and artifacts

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                            DATA FLOW DIAGRAM                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

    ┌──────────────┐
    │ flwr run     │  ← You execute this command
    │ local-deploy │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ SuperLink    │  ← Receives deployment request
    │ Port 9093    │
    └──────┬───────┘
           │
           ├─────────────────────────┬─────────────────────────┐
           ▼                         ▼                         ▼
    ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
    │ SuperNode-1 │          │ SuperNode-2 │          │ SuperNode-3 │
    │ Port 9094   │          │ Port 9095   │          │ Port 9096   │
    └──────┬──────┘          └──────┬──────┘          └──────┬──────┘
           │                        │                        │
           ▼                        ▼                        ▼
    ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
    │ ClientApp-1 │          │ ClientApp-2 │          │ ClientApp-3 │
    │ Partition 0 │          │ Partition 1 │          │ Partition 2 │
    └──────┬──────┘          └──────┬──────┘          └──────┬──────┘
           │                        │                        │
           └────────────┬───────────┴────────────┬───────────┘
                        │                        │
                        ▼                        ▼
                 ┌──────────────┐        ┌──────────────┐
                 │  ServerApp   │        │   MLflow     │
                 │  Aggregates  │───────►│   Logs All   │
                 │  Updates     │        │   Metrics    │
                 └──────────────┘        └──────────────┘

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                          NETWORK COMMUNICATION                               ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

    Protocol: gRPC (Flower's default)
    Network: Docker bridge network (complete_default)
    
    Communication Paths:
    ────────────────────
    
    1. Command Execution:
       Host → SuperLink:9093 (flwr run command)
    
    2. Container → Container:
       SuperLink ↔ SuperNode-1 (internal)
       SuperLink ↔ SuperNode-2 (internal)
       SuperLink ↔ SuperNode-3 (internal)
       SuperNode-1 ↔ ClientApp-1 (internal)
       SuperNode-2 ↔ ClientApp-2 (internal)
       SuperNode-3 ↔ ClientApp-3 (internal)
    
    3. Logging:
       ServerApp → MLflow:5000 (HTTP)
       ClientApp-1 → MLflow:5000 (HTTP)
       ClientApp-2 → MLflow:5000 (HTTP)
       ClientApp-3 → MLflow:5000 (HTTP)
    
    4. Monitoring:
       Platform UI → Docker Socket (read-only)
       Browser → Platform UI:8050 (HTTP)
       Browser → MLflow:5000 (HTTP)
       Browser → SuperLink:9093 (HTTP/gRPC)

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                            KEY TECHNOLOGIES                                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

    • Flower Framework      → Federated Learning orchestration
    • PyTorch              → Deep learning framework
    • Docker & Compose     → Containerization and orchestration
    • MLflow               → Experiment tracking and model registry
    • Dash/Flask           → Web-based monitoring UI
    • gRPC                 → Efficient client-server communication
    • MedMNIST             → Medical image dataset (pneumonia detection)
    • Python 3.12          → Programming language

    Algorithms:
    • FedAvg               → Federated Averaging (default aggregation)
    • IID Partitioning     → Data distribution strategy
    • SGD                  → Stochastic Gradient Descent (client-side)

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                              FILE STRUCTURE                                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

    Federated_Learning/
    │
    ├── complete/
    │   ├── compose-with-ui.yml      ← Docker Compose config (with UI & MLflow)
    │   ├── compose.yml              ← Basic Docker Compose config
    │   └── fl/                      ← Flower App
    │       ├── fl/
    │       │   ├── server_app.py    ← Server FL logic + MLflow
    │       │   ├── client_app.py    ← Client FL logic + MLflow
    │       │   ├── task.py          ← Model and training functions
    │       │   ├── tracking.py      ← MLflow utilities
    │       │   ├── partitioning.py  ← Data partitioning (FIXED)
    │       │   └── config.py        ← Config management
    │       ├── config/
    │       │   └── default.yaml     ← Training config
    │       └── pyproject.toml       ← Flower app config
    │
    ├── platform-ui/
    │   └── app.py                   ← Monitoring dashboard
    │
    ├── SUMMARY.md                   ← Quick overview (START HERE)
    ├── RUNNING_GUIDE.md             ← Complete instructions
    ├── QUICK_REFERENCE.md           ← Command reference
    ├── MLFLOW_GUIDE.md              ← MLflow usage guide
    ├── ARCHITECTURE.md              ← This file
    └── README.md                    ← Main documentation

```

## 🎯 Quick Understanding

**What it does**: Trains a machine learning model across 3 separate Docker containers (clients) without sharing data, while tracking everything in MLflow.

**How it works**: 
1. Server creates a model
2. Sends to clients
3. Each client trains on their data
4. Clients send updates back
5. Server averages the updates
6. Repeat for 3 rounds
7. MLflow tracks all metrics

**Why Docker containers**: 
- Each client is isolated (like real-world federated learning)
- Easy to scale (add more clients)
- Production-ready architecture

**Why MLflow**:
- Track experiments automatically
- Compare different runs
- Reproducible results
- Easy analysis and export

## 🚀 Next Steps

1. **Read**: [SUMMARY.md](SUMMARY.md)
2. **Run**: `flwr run fl local-deployment --stream`
3. **Monitor**: http://localhost:8050 and http://localhost:5000
4. **Analyze**: Compare client performance in MLflow
5. **Experiment**: Change hyperparameters and run again!
