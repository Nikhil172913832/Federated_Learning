#!/bin/bash

# Federated Learning Platform Showcase Script
# This script demonstrates the complete platform with Docker container clients

set -e

echo "🎯 Federated Learning Platform Showcase"
echo "======================================="
echo ""
echo "This showcase demonstrates:"
echo "  🐳 Docker container clients (3 separate containers)"
echo "  🖥️  Server in Docker container"
echo "  📊 Real-time monitoring UI"
echo "  📈 MLflow experiment tracking"
echo "  🌐 SuperLink coordination service"
echo ""

# Check if platform is running
echo "🔍 Checking if platform is running..."
if docker ps --format 'table {{.Names}}' | grep -q "fl-platform-ui"; then
    echo "✅ Platform is already running!"
    echo ""
    echo "🌐 Access Points:"
    echo "  📊 Platform Dashboard: http://localhost:8050"
    echo "  🌐 SuperLink API:      http://localhost:9093"
    echo "  📈 MLflow Tracking:    http://localhost:5000"
    echo ""
    echo "🎮 Ready to showcase federated learning!"
else
    echo "⚠️  Platform is not running. Starting it now..."
    echo ""
    ./launch-platform.sh
fi

echo ""
echo "📋 Showcase Steps:"
echo ""
echo "1. 🌐 Open Platform Dashboard"
echo "   URL: http://localhost:8050"
echo "   - View real-time container status"
echo "   - Monitor system resources"
echo "   - See client container details"
echo ""

echo "2. 🐳 Verify Docker Container Clients"
echo "   Run: docker ps"
echo "   You should see:"
echo "   - superlink (coordination service)"
echo "   - superexec-serverapp (server container)"
echo "   - supernode-1, supernode-2, supernode-3 (client nodes)"
echo "   - superexec-clientapp-1, superexec-clientapp-2, superexec-clientapp-3 (client containers)"
echo "   - fl-platform-ui (monitoring dashboard)"
echo "   - mlflow (experiment tracking)"
echo ""

echo "3. 🎯 Start Federated Learning"
echo "   Run: flwr run quickstart-compose local-deployment --stream"
echo "   This will:"
echo "   - Connect to SuperLink coordination service"
echo "   - Start federated learning rounds"
echo "   - Each client runs in its own Docker container"
echo "   - Server runs in its own Docker container"
echo ""

echo "4. 📊 Monitor in Real-time"
echo "   - Watch the Platform Dashboard (http://localhost:8050)"
echo "   - View container logs: docker compose -f complete/compose-with-ui.yml logs -f"
echo "   - Check MLflow experiments (http://localhost:5000)"
echo ""

echo "5. 🔍 Inspect Container Clients"
echo "   View client logs:"
echo "   - docker logs superexec-clientapp-1"
echo "   - docker logs superexec-clientapp-2"
echo "   - docker logs superexec-clientapp-3"
echo "   - docker logs superexec-serverapp"
echo ""

echo "🎉 Platform Showcase Ready!"
echo ""
echo "Key Features Demonstrated:"
echo "  ✅ Each client is a separate Docker container"
echo "  ✅ Server runs in Docker container"
echo "  ✅ Real-time monitoring UI"
echo "  ✅ Production-ready architecture"
echo "  ✅ Scalable design (easy to add more clients)"
echo "  ✅ Complete federated learning stack"
echo ""
echo "🚀 Start the showcase by opening http://localhost:8050 in your browser!"
