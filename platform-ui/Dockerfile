# Optimized Dockerfile for Platform UI
# Uses layer caching to avoid re-downloading dependencies on every build

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (cached layer)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements file first (for better caching)
# Create a requirements.txt for explicit dependency management
COPY platform-ui/requirements.txt /app/

# Install Python dependencies (cached layer)
# This only re-runs if requirements.txt changes
# Add retries and increased timeout for network issues
# Use Tsinghua University PyPI mirror for faster downloads
RUN pip install --default-timeout=100 --retries 5 --no-cache-dir \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -r requirements.txt

# Copy application code (separate layer - changes frequently)
# Code changes don't invalidate the dependency cache above
COPY platform-ui/*.py /app/

EXPOSE 8050

CMD ["python", "app.py"]
