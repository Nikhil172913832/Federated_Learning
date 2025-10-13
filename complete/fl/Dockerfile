# Optimized Dockerfile for FL Training (Server and Client containers)
# Uses layer caching to avoid re-downloading dependencies on every build

FROM flwr/superexec:1.22.0

# Install system dependencies (cached layer)
USER root
RUN apt-get update \
    && apt-get -y --no-install-recommends install \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
USER app

WORKDIR /app

# Copy only dependency files first (for better caching)
# This layer is cached unless pyproject.toml changes
COPY --chown=app:app complete/fl/pyproject.toml /app/

# Install Python dependencies (cached layer)
# This only re-runs if pyproject.toml changes
# Add retries and increased timeout for network issues
# Use Tsinghua University PyPI mirror for faster downloads
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
    && python -m pip install --default-timeout=100 --retries 5 --no-cache-dir \
       --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
       --trusted-host pypi.tuna.tsinghua.edu.cn \
       -e .

# Copy application code (separate layer - changes frequently)
# Code changes don't invalidate the dependency cache above
COPY --chown=app:app complete/fl/fl/ /app/fl/
COPY --chown=app:app complete/fl/config/ /app/config/
COPY --chown=app:app complete/fl/scripts/ /app/scripts/
COPY --chown=app:app complete/fl/tests/ /app/tests/
COPY --chown=app:app complete/fl/README.md /app/

ENTRYPOINT ["flower-superexec"]
