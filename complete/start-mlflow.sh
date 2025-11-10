#!/bin/bash
# Script to inject dark mode CSS into MLflow UI

MLFLOW_STATIC_DIR="/usr/local/lib/python3.10/site-packages/mlflow/server/js/build/static"
CSS_FILE="/app/mlflow-dark.css"

# Wait for MLflow to be installed
sleep 2

# Find the main CSS file
if [ -d "$MLFLOW_STATIC_DIR/css" ]; then
    MAIN_CSS=$(find "$MLFLOW_STATIC_DIR/css" -name "*.css" | head -1)
    
    if [ -f "$MAIN_CSS" ] && [ -f "$CSS_FILE" ]; then
        echo "Injecting dark mode CSS into MLflow..."
        cat "$CSS_FILE" >> "$MAIN_CSS"
        echo "Dark mode CSS injected successfully!"
    else
        echo "Warning: Could not find CSS files to inject dark mode"
    fi
else
    echo "Warning: MLflow static directory not found"
fi

# Start MLflow server
exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root /app/mlruns
