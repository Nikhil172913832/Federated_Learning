#!/bin/bash

# Script to clean MLflow data only (keeps containers running)
# Usage: ./clean-mlflow.sh

echo "🧹 Cleaning MLflow data..."

# Get the MLflow container ID
MLFLOW_CONTAINER=$(docker ps -q -f name=complete-mlflow)

if [ -n "$MLFLOW_CONTAINER" ]; then
    echo "📦 Found MLflow container: $MLFLOW_CONTAINER"
    
    # Stop MLflow container
    docker stop $MLFLOW_CONTAINER
    
    # Remove MLflow volume
    docker volume rm complete_mlflow-data 2>/dev/null || echo "Volume already removed"
    
    # Restart MLflow container
    docker compose -f compose-with-ui.yml up -d mlflow
    
    echo "✅ MLflow data cleaned and container restarted!"
    echo "📊 MLflow UI: http://localhost:5000"
else
    echo "⚠️  MLflow container not found. Starting fresh..."
    docker volume rm complete_mlflow-data 2>/dev/null || true
    docker compose -f compose-with-ui.yml up -d mlflow
    echo "✅ MLflow started fresh!"
fi
