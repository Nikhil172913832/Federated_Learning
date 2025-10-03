# Federated Learning (Flower)

This repository provides an end-to-end, federated learning platform using the Flower framework and PyTorch. It includes:
- Config-driven experiments (`complete/fl/config/default.yaml`)
- Non-IID client heterogeneity
- Personalization (FedProx)
- Differential Privacy (DP-SGD via Opacus)
- Per-client storage simulation (folder/SQLite)
- MLflow tracking and per-round checkpoints
- Unit tests, GitHub Actions CI, and a Makefile
- Docker Compose setup for Flower Deployment Engine

See `complete/fl/README.md` for detailed instructions and quickstart.
