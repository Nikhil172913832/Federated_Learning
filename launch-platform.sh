#!/bin/bash

# Federated Learning Platform Launcher
# This script launches a complete federated learning platform with Docker container clients and monitoring UI

set -e

echo "🚀 Federated Learning Platform Launcher"
echo "========================================"
echo ""
echo "This will launch a complete federated learning platform featuring:"
echo "  🐳 Docker container clients (3 clients)"
echo "  🖥️  Server in Docker container"
echo "  🌐 SuperLink coordination service"
echo "  📊 Real-time monitoring UI"
echo "  📈 MLflow tracking server"
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    exit 1
fi
echo "✅ Docker is running"

# Check Docker Compose
if ! command -v docker compose > /dev/null 2>&1; then
    echo "❌ Error: Docker Compose V2 is not installed"
    exit 1
fi
echo "✅ Docker Compose V2 is available"

# Check if complete directory exists
if [ ! -d "complete" ]; then
    echo "❌ Error: complete directory not found. Run setup-docker-compose.sh first"
    exit 1
fi
echo "✅ Complete directory exists"

# Check if platform-ui directory exists
if [ ! -d "platform-ui" ]; then
    echo "❌ Error: platform-ui directory not found"
    exit 1
fi
echo "✅ Platform UI directory exists"

# Set PROJECT_DIR
if [ -z "$PROJECT_DIR" ]; then
    export PROJECT_DIR="quickstart-compose"
    echo "⚠️  PROJECT_DIR not set, using default: $PROJECT_DIR"
fi
echo "✅ PROJECT_DIR set to: $PROJECT_DIR"

echo ""
echo "🏗️  Step 1: Building and starting the platform..."
echo "This includes:"
echo "  - SuperLink (coordination service)"
echo "  - SuperExec-ServerApp (server container)"
echo "  - 3 SuperNodes (client nodes)"
echo "  - 3 SuperExec-ClientApps (client containers)"
echo "  - Platform UI (monitoring dashboard)"
echo "  - MLflow tracking server"
echo ""

cd complete

# Stop any existing services
echo "🛑 Stopping any existing services..."
docker compose -f compose-with-ui.yml down 2>/dev/null || true

# Start the platform
echo "🚀 Starting federated learning platform..."
export PROJECT_DIR=$PROJECT_DIR
docker compose -f compose-with-ui.yml up --build -d

echo ""
echo "⏳ Step 2: Waiting for services to be ready..."
sleep 45

echo ""
echo "📊 Step 3: Checking platform status..."
echo "Active services:"
docker compose -f compose-with-ui.yml ps

echo ""
echo "🔍 Step 4: Verifying platform components..."

# Check SuperLink
echo "🌐 Checking SuperLink..."
if curl -f http://localhost:9093/health > /dev/null 2>&1; then
    echo "✅ SuperLink is accessible at http://localhost:9093"
else
    echo "⚠️  SuperLink health check failed"
fi

# Check Platform UI
echo "📊 Checking Platform UI..."
if curl -f http://localhost:8050 > /dev/null 2>&1; then
    echo "✅ Platform UI is accessible at http://localhost:8050"
else
    echo "⚠️  Platform UI not yet ready"
fi

# Check MLflow
echo "📈 Checking MLflow..."
if curl -f http://localhost:5000 > /dev/null 2>&1; then
    echo "✅ MLflow is accessible at http://localhost:5000"
else
    echo "⚠️  MLflow not yet ready"
fi

echo ""
echo "🎯 Step 5: Platform is ready!"
echo ""
echo "🌐 Access Points:"
echo "  📊 Platform Dashboard: http://localhost:8050"
echo "  🌐 SuperLink API:      http://localhost:9093"
echo "  📈 MLflow Tracking:    http://localhost:5000"
echo ""
echo "🐳 Container Clients:"
echo "  - SuperExec-ClientApp-1 (Client 1)"
echo "  - SuperExec-ClientApp-2 (Client 2)"
echo "  - SuperExec-ClientApp-3 (Client 3)"
echo "  - SuperExec-ServerApp (Server)"
echo ""
echo "🎮 Next Steps:"
echo "  1. Open http://localhost:8050 in your browser to see the platform dashboard"
echo "  2. Monitor real-time container status and system resources"
echo "  3. Start federated learning by running:"
echo "     cd .."
echo "     flwr run $PROJECT_DIR local-deployment --stream"
echo ""
echo "📋 Useful Commands:"
echo "  View logs:           docker compose -f compose-with-ui.yml logs -f"
echo "  Stop platform:       docker compose -f compose-with-ui.yml down"
echo "  Restart platform:    docker compose -f compose-with-ui.yml restart"
echo "  View containers:     docker compose -f compose-with-ui.yml ps"
echo ""
echo "🎉 Federated Learning Platform is now running!"
echo "   Each client is a separate Docker container as requested!"
