#!/bin/bash
# Auto-format code with Black and isort

set -e

echo "🎨 Formatting code..."
echo ""

# Format with Black
echo "📝 Formatting with Black..."
if command -v black &> /dev/null; then
    black complete/fl/fl/
    echo "✅ Black formatting complete"
else
    echo "⚠️  Black not installed. Install with: pip install black"
fi
echo ""

# Sort imports with isort
echo "📦 Sorting imports with isort..."
if command -v isort &> /dev/null; then
    isort complete/fl/fl/
    echo "✅ Import sorting complete"
else
    echo "⚠️  isort not installed. Install with: pip install isort"
fi
echo ""

echo "✅ Code formatting complete!"
