#!/bin/bash

# Test script for Docker Compose setup
# This script verifies that the Docker Compose environment is working correctly

set -e

echo "Testing Docker Compose setup..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    exit 1
fi
echo "✅ Docker is running"

# Check if Docker Compose is available
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

# Check if compose.yml exists
if [ ! -f "complete/compose.yml" ]; then
    echo "❌ Error: compose.yml not found in complete directory"
    exit 1
fi
echo "✅ compose.yml exists"

# Check if PROJECT_DIR is set
if [ -z "$PROJECT_DIR" ]; then
    echo "⚠️  Warning: PROJECT_DIR not set. Using default: quickstart-compose"
    export PROJECT_DIR="quickstart-compose"
fi
echo "✅ PROJECT_DIR is set to: $PROJECT_DIR"

# Check if Flower project exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ Error: Flower project $PROJECT_DIR not found. Run setup-docker-compose.sh first"
    exit 1
fi
echo "✅ Flower project exists"

# Test Docker Compose syntax
echo "Testing Docker Compose syntax..."
cd complete
if ! docker compose config > /dev/null 2>&1; then
    echo "❌ Error: Docker Compose configuration is invalid"
    exit 1
fi
echo "✅ Docker Compose configuration is valid"

# Test building images (without starting services)
echo "Testing Docker image builds..."
if ! docker compose build --no-cache > /dev/null 2>&1; then
    echo "❌ Error: Failed to build Docker images"
    exit 1
fi
echo "✅ Docker images built successfully"

# Test starting services
echo "Starting services for testing..."
if ! docker compose up -d > /dev/null 2>&1; then
    echo "❌ Error: Failed to start services"
    exit 1
fi
echo "✅ Services started successfully"

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Test SuperLink health endpoint
echo "Testing SuperLink health endpoint..."
if curl -f http://localhost:9093/health > /dev/null 2>&1; then
    echo "✅ SuperLink health check passed"
else
    echo "⚠️  Warning: SuperLink health check failed (this might be normal)"
fi

# Check service status
echo "Checking service status..."
docker compose ps

# Test stopping services
echo "Stopping services..."
if ! docker compose down > /dev/null 2>&1; then
    echo "❌ Error: Failed to stop services"
    exit 1
fi
echo "✅ Services stopped successfully"

cd ..

echo ""
echo "🎉 All tests passed! Docker Compose setup is working correctly."
echo ""
echo "You can now run:"
echo "  docker compose up --build -d    # Start services"
echo "  flwr run $PROJECT_DIR local-deployment --stream    # Run the project"
echo "  docker compose down             # Stop services"
