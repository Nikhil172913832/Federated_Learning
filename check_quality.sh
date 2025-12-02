#!/bin/bash
# Code Quality Check Script
# Run all linting and formatting checks

set -e

echo "🔍 Running code quality checks..."
echo ""

# Black formatting check
echo "📝 Checking code formatting with Black..."
if command -v black &> /dev/null; then
    black --check --diff complete/fl/fl/ || true
else
    echo "⚠️  Black not installed. Install with: pip install black"
fi
echo ""

# isort import sorting check
echo "📦 Checking import sorting with isort..."
if command -v isort &> /dev/null; then
    isort --check-only --diff complete/fl/fl/ || true
else
    echo "⚠️  isort not installed. Install with: pip install isort"
fi
echo ""

# Flake8 linting
echo "🔎 Linting with Flake8..."
if command -v flake8 &> /dev/null; then
    flake8 complete/fl/fl/ || true
else
    echo "⚠️  Flake8 not installed. Install with: pip install flake8"
fi
echo ""

# MyPy type checking
echo "🏷️  Type checking with MyPy..."
if command -v mypy &> /dev/null; then
    mypy complete/fl/fl/ --ignore-missing-imports || true
else
    echo "⚠️  MyPy not installed. Install with: pip install mypy"
fi
echo ""

# Bandit security check
echo "🔒 Security check with Bandit..."
if command -v bandit &> /dev/null; then
    bandit -r complete/fl/fl/ -ll || true
else
    echo "⚠️  Bandit not installed. Install with: pip install bandit"
fi
echo ""

# Pytest
echo "🧪 Running tests with pytest..."
if command -v pytest &> /dev/null; then
    pytest complete/fl/tests/ -v || true
else
    echo "⚠️  Pytest not installed. Install with: pip install pytest"
fi
echo ""

echo "✅ Code quality checks complete!"
echo ""
echo "To auto-fix formatting issues, run:"
echo "  black complete/fl/fl/"
echo "  isort complete/fl/fl/"
