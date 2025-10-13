#!/bin/bash

# Quick Start Script - FL Platform without UI Container
# This starts the FL platform and UI separately to avoid Docker build issues

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  🚀 Federated Learning Platform - Quick Start           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    exit 1
fi
echo "✅ Docker is running"

echo ""
echo "📦 Step 1: Starting FL containers (without UI)..."
cd complete
docker compose -f compose.yml up -d

echo ""
echo "⏳ Step 2: Waiting for services to be ready (30 seconds)..."
sleep 30

echo ""
echo "✅ Step 3: FL Platform is ready!"
echo ""
echo "🌐 Access Points:"
echo "  🌐 SuperLink API:      http://localhost:9093"
echo ""
echo "🐳 Running Containers:"
docker compose -f compose.yml ps
echo ""
echo "🎯 Next Steps:"
echo ""
echo "  Option 1 - Start Training from Terminal:"
echo "    cd complete"
echo "    flwr run fl local-deployment --stream"
echo ""
echo "  Option 2 - Start the Platform UI (in another terminal):"
echo "    cd platform-ui"
echo "    pip install -r requirements.txt  # Only needed first time"
echo "    python3 app.py"
echo "    # Then open http://localhost:8050"
echo ""
echo "  Option 3 - Just run training without UI:"
echo "    cd complete"
echo "    flwr run fl local-deployment --stream"
echo ""
echo "💡 Tip: The UI provides easy start/stop and monitoring,"
echo "    but you can run training from terminal too!"
echo ""
echo "✅ Platform ready! Choose your preferred method above."
