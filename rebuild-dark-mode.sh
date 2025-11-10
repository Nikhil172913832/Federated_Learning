#!/bin/bash
# Rebuild script for the Federated Learning Platform

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Federated Learning Platform - Rebuild"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Navigate to complete directory
cd "$(dirname "$0")/complete"

echo "Stopping existing containers..."
docker compose -f compose-with-ui.yml down

echo ""
echo "Rebuilding containers..."
echo "   - MLflow tracking server"
echo "   - Dashboard UI"
echo "   - Enhanced components"
echo ""

docker compose -f compose-with-ui.yml build --no-cache mlflow fl-platform-ui

echo ""
echo "Starting containers..."
docker compose -f compose-with-ui.yml up -d

echo ""
echo "Waiting for services to be ready..."
sleep 5

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Deployment Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Access Points:"
echo "   Dashboard: http://localhost:8050"
echo "   MLflow:    http://localhost:5000"
echo "   SuperLink: http://localhost:9093"
echo ""
echo "Tips:"
echo "   - Use the dashboard to start/stop training"
echo "   - Check logs: docker compose -f compose-with-ui.yml logs -f"
echo "   - View containers: docker ps"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
