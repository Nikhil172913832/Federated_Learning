# Optimized Dockerfile for MLflow with Dark Mode
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

# Copy dark mode CSS and startup script
COPY complete/mlflow-dark.css /app/mlflow-dark.css
COPY complete/start-mlflow.sh /app/start-mlflow.sh

# Make startup script executable
RUN chmod +x /app/start-mlflow.sh

EXPOSE 5000

# Use custom startup script that injects dark mode
CMD ["/app/start-mlflow.sh"]
