#!/bin/bash

# Script to run federated learning with Docker containers as clients
# This follows the official Flower documentation for containerized clients

set -e

echo "🚀 Running Federated Learning with Docker Container Clients"
echo "=========================================================="

# Check if we're in the right directory
if [ ! -d "complete" ]; then
    echo "❌ Error: complete directory not found. Run setup-docker-compose.sh first"
    exit 1
fi

# Check if PROJECT_DIR is set
if [ -z "$PROJECT_DIR" ]; then
    echo "⚠️  Warning: PROJECT_DIR not set. Using default: quickstart-compose"
    export PROJECT_DIR="quickstart-compose"
fi

echo "📋 Configuration:"
echo "  - PROJECT_DIR: $PROJECT_DIR"
echo "  - Client containers: 2 SuperNodes + 2 SuperExec ClientApps"
echo "  - Server container: 1 SuperExec ServerApp"
echo "  - Coordination: 1 SuperLink"
echo ""

# Navigate to complete directory
cd complete

echo "🔧 Step 1: Building and starting all services..."
echo "This includes:"
echo "  - SuperLink (coordination service)"
echo "  - SuperNode-1 & SuperNode-2 (client nodes)"
echo "  - SuperExec-ServerApp (server container)"
echo "  - SuperExec-ClientApp-1 & SuperExec-ClientApp-2 (client containers)"
echo ""

export PROJECT_DIR=$PROJECT_DIR
docker compose up --build -d

echo ""
echo "⏳ Step 2: Waiting for services to be ready..."
sleep 30

echo ""
echo "📊 Step 3: Checking service status..."
echo "Active containers:"
docker compose ps

echo ""
echo "🔍 Step 4: Verifying client containers are running..."
CLIENT_CONTAINERS=$(docker compose ps --services | grep -E "(supernode|superexec-clientapp)" | wc -l)
echo "Found $CLIENT_CONTAINERS client-related containers"

if [ $CLIENT_CONTAINERS -ge 4 ]; then
    echo "✅ Client containers are running successfully!"
else
    echo "⚠️  Warning: Expected at least 4 client containers, found $CLIENT_CONTAINERS"
fi

echo ""
echo "🌐 Step 5: Testing SuperLink connectivity..."
if curl -f http://localhost:9093/health > /dev/null 2>&1; then
    echo "✅ SuperLink is accessible"
else
    echo "⚠️  SuperLink health check failed (this might be normal)"
fi

echo ""
echo "🎯 Step 6: Running federated learning with containerized clients..."
echo "This will start the federated learning process where:"
echo "  - Server runs in a Docker container (SuperExec-ServerApp)"
echo "  - Clients run in separate Docker containers (SuperExec-ClientApp-1, SuperExec-ClientApp-2)"
echo "  - All communication happens through the SuperLink coordination service"
echo ""

# Go back to parent directory to run flwr command
cd ..

echo "🚀 Starting federated learning..."
echo "Command: flwr run $PROJECT_DIR local-deployment --stream"
echo ""
echo "Press Ctrl+C to stop the federated learning process"
echo ""

flwr run $PROJECT_DIR local-deployment --stream

echo ""
echo "🏁 Federated learning completed!"
echo ""
echo "📋 To view logs from client containers:"
echo "  docker compose logs superexec-clientapp-1"
echo "  docker compose logs superexec-clientapp-2"
echo "  docker compose logs superexec-serverapp"
echo ""
echo "🛑 To stop all services:"
echo "  docker compose down"
echo ""
echo "📊 To view all container logs:"
echo "  docker compose logs -f"
