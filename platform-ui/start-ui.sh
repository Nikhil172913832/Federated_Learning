#!/bin/bash

# Quick Start Script for Federated Learning Platform UI
# This script sets up and launches the enhanced FL Platform UI

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  🚀 Federated Learning Platform UI - Quick Start        ║"
echo "╔══════════════════════════════════════════════════════════╗"
echo ""

# Check if we're in the platform-ui directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: Please run this script from the platform-ui directory"
    exit 1
fi

# Check Python
if ! command -v python3 > /dev/null 2>&1; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi
echo "✅ Python 3 found"

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    exit 1
fi
echo "✅ Docker is running"

# Install dependencies
echo ""
echo "📦 Installing Python dependencies..."
    pip install -r requirements.txt -q

echo ""
echo "✨ Enhanced UI Features:"
echo "   • Start/Stop FL training from browser"
echo "   • Configure training parameters"
echo "   • Real-time log viewing"
echo "   • System monitoring"
echo "   • Container status tracking"
echo ""

echo "🌐 Starting Platform UI..."
echo ""
echo "📊 Dashboard will be available at: http://localhost:8050"
echo "📈 MLflow UI will be at: http://localhost:5000"
echo ""
echo "💡 Tip: Use the 'Start Training' button in the UI to begin FL training"
echo ""

# Start the application
python3 app.py
