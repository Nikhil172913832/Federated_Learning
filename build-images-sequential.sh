#!/bin/bash

# Sequential Docker Build Script
# Builds images one at a time to prevent system crashes from parallel OCI exports

set -e

echo "=========================================="
echo "Sequential Docker Image Build"
echo "=========================================="
echo ""
echo "This script builds images sequentially to avoid overwhelming the system."
echo "Building 6 images one at a time may take longer but is more stable."
echo ""

cd "$(dirname "$0")/complete"

# Build images sequentially
echo "Step 1: Building images sequentially..."
echo ""

# Build 1: MLflow
echo "[1/6] Building MLflow tracking server..."
docker build -f mlflow.Dockerfile -t complete-mlflow:latest .. || { echo "Failed to build MLflow"; exit 1; }
echo "✓ MLflow built successfully"
echo ""

# Build 2: Platform UI
echo "[2/6] Building Platform UI..."
docker build -f ../platform-ui/Dockerfile -t complete-fl-platform-ui:latest .. || { echo "Failed to build Platform UI"; exit 1; }
echo "✓ Platform UI built successfully"
echo ""

# Build 3: ServerApp
echo "[3/6] Building SuperExec ServerApp..."
docker build -f fl/${FL_DOCKERFILE:-Dockerfile.cpu} -t complete-superexec-serverapp:latest .. || { echo "Failed to build ServerApp"; exit 1; }
echo "✓ ServerApp built successfully"
echo ""

# Build 4-6: ClientApps (same image, used by 3 services)
echo "[4/6] Building SuperExec ClientApp..."
echo "Note: This image will be shared by all 3 client containers"
docker build -f fl/${FL_DOCKERFILE:-Dockerfile.cpu} -t complete-superexec-clientapp:latest .. || { echo "Failed to build ClientApp"; exit 1; }
echo "✓ ClientApp built successfully"
echo ""

echo "=========================================="
echo "All images built successfully!"
echo "=========================================="
echo ""
echo "Built images:"
docker images | grep -E "complete-|REPOSITORY" | head -10
echo ""
echo "Next steps:"
echo "1. The images are now built and cached"
echo "2. Run: ./launch-platform.sh"
echo "   (It will use the pre-built images and start quickly)"
echo ""
