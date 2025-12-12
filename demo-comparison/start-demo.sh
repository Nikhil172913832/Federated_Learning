#!/bin/bash
# Quick start script for the demo

echo "🏥 Pneumonia Detection Demo - Quick Start"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt --quiet

# Run the app
echo ""
echo "🚀 Starting the demo application..."
echo ""
python app.py
