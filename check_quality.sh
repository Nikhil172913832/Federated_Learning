#!/bin/bash
# Code quality checks for interview readiness

set -e

echo "=========================================="
echo "Running Code Quality Checks"
echo "=========================================="
echo ""

# Check if in correct directory
if [ ! -d "core_fl" ]; then
    echo "Error: Must run from project root"
    exit 1
fi

echo "1. Running Black formatter check..."
python -m black --check core_fl/ examples/ tests/ || {
    echo "❌ Black formatting failed"
    echo "Run: python -m black core_fl/ examples/ tests/"
    exit 1
}
echo "✅ Black formatting passed"
echo ""

echo "2. Running Flake8 linter..."
python -m flake8 core_fl/ examples/ tests/ --max-line-length=100 --ignore=E203,W503 || {
    echo "❌ Flake8 linting failed"
    exit 1
}
echo "✅ Flake8 linting passed"
echo ""

echo "3. Running MyPy type checker..."
python -m mypy core_fl/ --ignore-missing-imports || {
    echo "⚠️  MyPy found type issues (non-blocking)"
}
echo "✅ MyPy type checking completed"
echo ""

echo "4. Running unit tests..."
python -m pytest tests/test_core_server.py tests/test_core_client.py -v || {
    echo "❌ Unit tests failed"
    exit 1
}
echo "✅ Unit tests passed"
echo ""

echo "=========================================="
echo "All Code Quality Checks Passed! ✅"
echo "=========================================="
