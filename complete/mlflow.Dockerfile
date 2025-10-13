# Optimized Dockerfile for MLflow
# Uses layer caching for faster rebuilds

FROM python:3.10-slim

WORKDIR /app

# Install MLflow (cached layer unless version changes)
# Add retries and increased timeout for network issues
# Use Tsinghua University PyPI mirror for faster downloads
RUN pip install --default-timeout=100 --retries 5 --no-cache-dir \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    mlflow==2.9.2

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "/app/mlruns"]
