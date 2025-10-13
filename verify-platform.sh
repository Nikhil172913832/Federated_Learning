#!/bin/bash

# Platform Verification Script
# Checks if all components are working correctly

echo "🔍 Federated Learning Platform Verification"
echo "==========================================="
echo ""

cd complete

# Check if services are running
echo "1️⃣  Checking running services..."
RUNNING=$(docker compose -f compose-with-ui.yml ps --services --filter "status=running" | wc -l)
echo "   Running services: $RUNNING/10"

if [ "$RUNNING" -lt 10 ]; then
    echo "   ⚠️  Warning: Not all services are running"
    echo "   Expected: 10 (superlink, 3 supernodes, 3 clientapps, 1 serverapp, mlflow, ui)"
else
    echo "   ✅ All services running"
fi

echo ""

# Check network
echo "2️⃣  Checking Docker network..."
NETWORK=$(docker network ls | grep "fl-network" | wc -l)
if [ "$NETWORK" -eq 1 ]; then
    echo "   ✅ fl-network exists"
    
    # Count containers on network
    CONTAINERS_ON_NET=$(docker network inspect complete_fl-network --format '{{len .Containers}}' 2>/dev/null || echo "0")
    echo "   Containers on network: $CONTAINERS_ON_NET"
else
    echo "   ❌ fl-network not found"
fi

echo ""

# Check SuperLink
echo "3️⃣  Checking SuperLink..."
SUPERLINK_STATUS=$(docker inspect complete-superlink-1 --format '{{.State.Status}}' 2>/dev/null || echo "not found")
if [ "$SUPERLINK_STATUS" == "running" ]; then
    echo "   ✅ SuperLink is running at http://localhost:9093 (gRPC)"
    # Check if port is listening
    if nc -z localhost 9093 2>/dev/null; then
        echo "   ✅ SuperLink gRPC port 9093 is accessible"
    else
        echo "   ⚠️  SuperLink port 9093 not accessible (may still be starting)"
    fi
else
    echo "   ❌ SuperLink is not running (status: $SUPERLINK_STATUS)"
fi

echo ""

# Check MLflow
echo "4️⃣  Checking MLflow..."
if curl -sf http://localhost:5000/health > /dev/null 2>&1; then
    echo "   ✅ MLflow is accessible at http://localhost:5000"
    
    # Check experiments
    EXPERIMENTS=$(curl -sf http://localhost:5000/api/2.0/mlflow/experiments/search 2>/dev/null | grep -o '"experiment_id"' | wc -l)
    echo "   Experiments: $EXPERIMENTS"
else
    echo "   ⚠️  MLflow not accessible"
fi

echo ""

# Check Platform UI
echo "5️⃣  Checking Platform UI..."
if curl -sf http://localhost:8050 > /dev/null 2>&1; then
    echo "   ✅ Platform UI is accessible at http://localhost:8050"
else
    echo "   ⚠️  Platform UI not accessible"
fi

echo ""

# Check MLflow connectivity from containers
echo "6️⃣  Checking MLflow connectivity from containers..."
if docker exec complete-superexec-serverapp-1 python -c "import socket; socket.create_connection(('mlflow', 5000), timeout=5)" > /dev/null 2>&1; then
    echo "   ✅ ServerApp can reach MLflow"
else
    echo "   ❌ ServerApp cannot reach MLflow"
fi

if docker exec complete-superexec-clientapp-1-1 python -c "import socket; socket.create_connection(('mlflow', 5000), timeout=5)" > /dev/null 2>&1; then
    echo "   ✅ ClientApp-1 can reach MLflow"
else
    echo "   ❌ ClientApp-1 cannot reach MLflow"
fi

echo ""

# Check environment variables
echo "7️⃣  Checking environment variables..."
SERVER_MLFLOW=$(docker exec complete-superexec-serverapp-1 env 2>/dev/null | grep MLFLOW_TRACKING_URI || echo "NOT SET")
if [[ "$SERVER_MLFLOW" == *"http://mlflow:5000"* ]]; then
    echo "   ✅ ServerApp MLFLOW_TRACKING_URI correctly set"
else
    echo "   ❌ ServerApp MLFLOW_TRACKING_URI: $SERVER_MLFLOW"
fi

CLIENT_MLFLOW=$(docker exec complete-superexec-clientapp-1-1 env 2>/dev/null | grep MLFLOW_TRACKING_URI || echo "NOT SET")
if [[ "$CLIENT_MLFLOW" == *"http://mlflow:5000"* ]]; then
    echo "   ✅ ClientApp MLFLOW_TRACKING_URI correctly set"
else
    echo "   ❌ ClientApp MLFLOW_TRACKING_URI: $CLIENT_MLFLOW"
fi

echo ""

# Summary
echo "📋 Summary"
echo "=========="
echo "✅ Ready to start training if all checks passed"
echo ""
echo "To start training:"
echo "  cd complete"
echo "  flwr run fl local-deployment --stream"
echo ""
echo "To view logs:"
echo "  Dashboard: http://localhost:8050 (click 'View Logs')"
echo "  MLflow: http://localhost:5000"
echo "  Container logs: docker compose -f compose-with-ui.yml logs -f"
echo ""
