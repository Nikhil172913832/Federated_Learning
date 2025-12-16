# Dockerfile for FL Training (Server and Client containers)

FROM flwr/superexec:1.22.0

# Install system dependencies
USER root
RUN apt-get update \
    && apt-get -y --no-install-recommends install \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
USER app

WORKDIR /app

# Copy dependency files
COPY --chown=app:app complete/fl/pyproject.toml /app/

# Install Python dependencies
RUN sed -i 's/.*flwr\[simulation\].*//' pyproject.toml \
    && python -m pip install --no-cache-dir -e .

# Copy application code
COPY --chown=app:app complete/fl/fl/ /app/fl/
COPY --chown=app:app complete/fl/config/ /app/config/
COPY --chown=app:app complete/fl/scripts/ /app/scripts/
COPY --chown=app:app complete/fl/tests/ /app/tests/
COPY --chown=app:app complete/fl/README.md /app/

ENTRYPOINT ["flower-superexec"]
