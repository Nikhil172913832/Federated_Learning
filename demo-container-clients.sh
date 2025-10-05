#!/bin/bash

# Demonstration script showing the difference between local simulation and Docker container clients

set -e

echo "🎯 Federated Learning: Local Simulation vs Docker Container Clients"
echo "=================================================================="
echo ""

echo "This demo will show you the difference between:"
echo "1. Local simulation (clients as processes)"
echo "2. Docker container clients (clients as separate containers)"
echo ""

# Function to show running processes/containers
show_running() {
    echo "📊 Currently running:"
    echo "  - Docker containers: $(docker ps --format 'table {{.Names}}' | grep -E '(super|flwr)' | wc -l) containers"
    echo "  - Python processes: $(ps aux | grep -E '(flwr|python.*fl)' | grep -v grep | wc -l) processes"
    echo ""
}

echo "🔍 Step 1: Check current state"
show_running

echo "🚀 Step 2: Running LOCAL SIMULATION (clients as processes)"
echo "Command: cd complete/fl && flwr run ."
echo ""
echo "This will run federated learning with clients as Python processes on your local machine."
echo "Press Ctrl+C to stop the simulation and continue to container demo."
echo ""

read -p "Press Enter to start local simulation demo (or Ctrl+C to skip)..."

cd complete/fl
timeout 30s flwr run . || echo "Local simulation demo completed (timeout or stopped)"
cd ../..

echo ""
echo "🔍 Step 3: Check state after local simulation"
show_running

echo ""
echo "🐳 Step 4: Running DOCKER CONTAINER CLIENTS"
echo "This will run federated learning with clients as separate Docker containers."
echo ""

read -p "Press Enter to start Docker container clients demo..."

echo "Starting Docker containers..."
cd complete
export PROJECT_DIR="quickstart-compose"
docker compose up --build -d

echo ""
echo "⏳ Waiting for containers to be ready..."
sleep 30

echo ""
echo "🔍 Step 5: Check running containers"
echo "Docker containers:"
docker compose ps

echo ""
echo "📊 Container details:"
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" | grep -E "(super|flwr|quickstart)"

echo ""
echo "🎯 Step 6: Running federated learning with container clients"
echo "Command: flwr run quickstart-compose local-deployment --stream"
echo ""
echo "This will run federated learning where:"
echo "  - Server runs in: SuperExec-ServerApp container"
echo "  - Client 1 runs in: SuperExec-ClientApp-1 container"
echo "  - Client 2 runs in: SuperExec-ClientApp-2 container"
echo "  - All communicate via: SuperLink container"
echo ""

read -p "Press Enter to start federated learning with container clients (or Ctrl+C to skip)..."

cd ..
timeout 60s flwr run quickstart-compose local-deployment --stream || echo "Container clients demo completed (timeout or stopped)"

echo ""
echo "🔍 Step 7: Final state check"
show_running

echo ""
echo "📋 Step 8: View container logs"
echo "Server container logs:"
docker compose logs --tail=10 superexec-serverapp

echo ""
echo "Client container logs:"
docker compose logs --tail=5 superexec-clientapp-1
docker compose logs --tail=5 superexec-clientapp-2

echo ""
echo "🛑 Step 9: Cleanup"
echo "Stopping all containers..."
docker compose down

echo ""
echo "✅ Demo completed!"
echo ""
echo "📊 Summary of differences:"
echo ""
echo "LOCAL SIMULATION:"
echo "  - Clients run as Python processes"
echo "  - All in same memory space"
echo "  - Faster startup"
echo "  - Less resource usage"
echo "  - Good for development/testing"
echo ""
echo "DOCKER CONTAINER CLIENTS:"
echo "  - Clients run in separate containers"
echo "  - True isolation between clients"
echo "  - Production-like environment"
echo "  - Higher resource usage"
echo "  - Better for realistic testing"
echo ""
echo "🎯 For production federated learning, use Docker container clients!"
echo "For quick development and testing, use local simulation."
