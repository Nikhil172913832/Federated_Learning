#!/bin/bash

# Script to start fresh training with clean MLflow logs
# Usage: ./fresh-training.sh

set -e

echo "🧹 Cleaning up old MLflow data..."

# Stop containers
docker compose -f compose-with-ui.yml down

# Remove MLflow volume to clear old logs
docker volume rm complete_mlflow-data 2>/dev/null || echo "MLflow volume already removed or doesn't exist"

# Start services
echo "🚀 Starting services with fresh MLflow..."
docker compose -f compose-with-ui.yml up -d

echo "✅ Services started with clean MLflow!"
echo "📊 MLflow UI: http://localhost:5000"
echo "🎯 Platform UI: http://localhost:8050"
echo ""
echo "Waiting for services to be ready..."
sleep 5

# Run training
echo "🏋️  Starting federated learning training..."
flwr run fl local-deployment --stream
