#!/bin/bash

# Setup script for Docker Compose with Flower framework
# This script sets up the complete Docker Compose environment

set -e

echo "Setting up Docker Compose environment for Flower framework..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker compose > /dev/null 2>&1; then
    echo "Error: Docker Compose V2 is not installed. Please install Docker Compose V2 and try again."
    exit 1
fi

# Check if flwr CLI is installed
if ! command -v flwr > /dev/null 2>&1; then
    echo "Installing flwr CLI..."
    pip install flwr
fi

# Clone the complete directory if it doesn't exist
if [ ! -d "complete" ]; then
    echo "Cloning Flower Docker Compose setup..."
    git clone --depth=1 --branch v1.22.0 https://github.com/adap/flower.git _tmp
    mv _tmp/framework/docker/complete .
    rm -rf _tmp
    echo "Docker Compose files cloned successfully."
else
    echo "Complete directory already exists. Skipping clone step."
fi

# Navigate to complete directory for Docker Compose operations
cd complete

# Create Flower project if it doesn't exist
PROJECT_DIR="quickstart-compose"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Creating Flower project: $PROJECT_DIR"
    flwr new $PROJECT_DIR --framework PyTorch --username flower
    echo "Flower project created successfully."
else
    echo "Flower project $PROJECT_DIR already exists. Skipping creation."
fi

# Set environment variable
export PROJECT_DIR=$PROJECT_DIR
echo "PROJECT_DIR set to: $PROJECT_DIR"

# Create .env file for convenience
echo "export PROJECT_DIR=$PROJECT_DIR" > .env
echo "Environment variables saved to .env file."

echo ""
echo "Setup complete! You can now run:"
echo "  docker compose up --build -d    # Start services"
echo "  flwr run $PROJECT_DIR local-deployment --stream    # Run the project"
echo "  docker compose down             # Stop services"
echo ""
echo "Or use the Makefile commands:"
echo "  make docker-up                  # Start services"
echo "  make docker-test                # Run full test"
echo "  make docker-down                # Stop services"
echo ""
echo "For advanced features:"
echo "  docker compose -f compose.yml -f with-state.yml up --build -d    # With state persistence"
echo "  docker compose -f compose.yml -f with-tls.yml up --build -d      # With TLS encryption"
echo "  docker compose -f certs.yml run --rm --build gen-certs           # Generate certificates"
