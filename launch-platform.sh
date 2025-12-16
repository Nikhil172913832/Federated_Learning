#!/bin/bash

# Federated Learning Platform Launcher

set -e

echo "Federated Learning Platform Launcher"
echo "========================================"
echo ""
echo "This will launch a complete federated learning platform:"
echo "  - SuperLink (coordination service)"
echo "  - 3 Client containers"
echo "  - Server container"
echo "  - Real-time monitoring UI"
echo "  - MLflow tracking server"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    exit 1
fi
echo "✓ Docker is running"

# Check Docker Compose
if ! command -v docker compose > /dev/null 2>&1; then
    echo "❌ Error: Docker Compose V2 is not installed"
    exit 1
fi
echo "✓ Docker Compose V2 is available"

# Check if complete directory exists
if [ ! -d "complete" ]; then
    echo "❌ Error: complete directory not found"
    exit 1
fi
echo "✓ Complete directory exists"

# Check if platform-ui directory exists
if [ ! -d "platform-ui" ]; then
    echo "❌ Error: platform-ui directory not found"
    exit 1
fi
echo "✓ Platform UI directory exists"

echo ""
echo "Building and starting the platform..."
echo "This will:"
echo "  1. Build fresh Docker images"
echo "  2. Start all services"
echo "  3. Initialize the platform"
echo ""
echo "⏱  Expected time: 5-10 minutes (first run)"
echo ""

cd complete

# Stop any existing services
echo "Stopping any existing services..."
docker compose -f compose-with-ui.yml down 2>/dev/null || true

# Build and start the platform (fresh build every time)
echo ""
echo "Building fresh images and starting services..."
docker compose -f compose-with-ui.yml up --build -d

echo ""
echo "Waiting for services to initialize..."
echo "This may take up to 60 seconds..."
sleep 60

echo ""
echo "Checking platform status..."
docker compose -f compose-with-ui.yml ps

echo ""
echo "✅ Platform is ready!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ACCESS POINTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  📊 Dashboard:  http://localhost:8050"
echo "  📈 MLflow:     http://localhost:5000"
echo "  🔗 SuperLink:  http://localhost:9093"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  NEXT STEPS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  1. Open http://localhost:8050 to see the dashboard"
echo "  2. Start federated learning:"
echo ""
echo "     cd complete"
echo "     flwr run fl local-deployment --stream"
echo ""
echo "  3. View experiment results at http://localhost:5000"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  USEFUL COMMANDS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  View logs:     docker compose -f compose-with-ui.yml logs -f"
echo "  Stop:          docker compose -f compose-with-ui.yml down"
echo "  Restart:       docker compose -f compose-with-ui.yml restart"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
