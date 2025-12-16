# Deployment Guide

This guide covers deploying the Federated Learning platform in production environments.

## Table of Contents
- [Local Deployment](#local-deployment)
- [Production Deployment](#production-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Security Considerations](#security-considerations)
- [Monitoring & Observability](#monitoring--observability)
- [Scaling Strategies](#scaling-strategies)

---

## Local Deployment

### Quick Start

```bash
# Clone repository
git clone https://github.com/Nikhil172913832/Federated_Learning.git
cd Federated_Learning

# Launch platform
./launch-platform.sh

# Verify
./verify-platform.sh
```

### Manual Deployment

```bash
# Start infrastructure
cd complete
docker compose -f compose-with-ui.yml up --build -d

# Check status
docker compose ps

# View logs
docker compose logs -f

# Access services
# - Dashboard: http://localhost:8050
# - MLflow: http://localhost:5000
# - SuperLink: http://localhost:9093
```

---

## Production Deployment

### Prerequisites

- Docker Engine 20.10+
- Docker Compose V2
- 16GB+ RAM
- 50GB+ disk space
- GPU (optional but recommended)

### Production Configuration

#### 1. Environment Variables

Create `.env` file:

```bash
# complete/.env
MLFLOW_TRACKING_URI=http://mlflow:5000
SUPERLINK_ADDRESS=superlink:9093
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1

# Security
SECRET_KEY=<generate-random-key>
ALLOWED_HOSTS=localhost,your-domain.com

# Resource limits
MAX_WORKERS=4
MEMORY_LIMIT=8g
CPU_LIMIT=4.0
```

#### 2. Production Compose File

```yaml
# complete/compose.prod.yml
version: '3.8'

services:
  superlink:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
      restart_policy:
        condition: on-failure
        max_attempts: 3
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  mlflow:
    environment:
      - MLFLOW_BACKEND_STORE_URI=postgresql://user:pass@db:5432/mlflow
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://mlflow-artifacts
    depends_on:
      - db

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=mlflow
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=mlflow
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

#### 3. Launch Production

```bash
cd complete
docker compose -f compose.yml -f compose.prod.yml up -d

# Check health
docker compose ps
docker compose logs -f
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes 1.24+
- kubectl configured
- Helm 3.0+ (optional)

### Deployment Manifests

#### 1. Namespace

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: federated-learning
```

#### 2. SuperLink Deployment

```yaml
# k8s/superlink-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: superlink
  namespace: federated-learning
spec:
  replicas: 1
  selector:
    matchLabels:
      app: superlink
  template:
    metadata:
      labels:
        app: superlink
    spec:
      containers:
      - name: superlink
        image: flwr/superlink:latest
        ports:
        - containerPort: 9093
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: superlink
  namespace: federated-learning
spec:
  selector:
    app: superlink
  ports:
  - port: 9093
    targetPort: 9093
  type: ClusterIP
```

#### 3. SuperNode Deployment

```yaml
# k8s/supernode-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: supernode
  namespace: federated-learning
spec:
  replicas: 10  # Number of clients
  selector:
    matchLabels:
      app: supernode
  template:
    metadata:
      labels:
        app: supernode
    spec:
      containers:
      - name: supernode
        image: your-registry/fl-client:latest
        env:
        - name: SUPERLINK_ADDRESS
          value: "superlink:9093"
        - name: PARTITION_ID
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

#### 4. MLflow Deployment

```yaml
# k8s/mlflow-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: federated-learning
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
      - name: mlflow
        image: your-registry/mlflow:latest
        ports:
        - containerPort: 5000
        env:
        - name: MLFLOW_BACKEND_STORE_URI
          value: "postgresql://mlflow:password@postgres:5432/mlflow"
        volumeMounts:
        - name: mlflow-artifacts
          mountPath: /mlflow/artifacts
      volumes:
      - name: mlflow-artifacts
        persistentVolumeClaim:
          claimName: mlflow-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow
  namespace: federated-learning
spec:
  selector:
    app: mlflow
  ports:
  - port: 5000
    targetPort: 5000
  type: LoadBalancer
```

#### 5. Deploy to Kubernetes

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Deploy components
kubectl apply -f k8s/superlink-deployment.yaml
kubectl apply -f k8s/supernode-deployment.yaml
kubectl apply -f k8s/mlflow-deployment.yaml

# Check status
kubectl get pods -n federated-learning
kubectl get services -n federated-learning

# View logs
kubectl logs -f deployment/superlink -n federated-learning
```

---

## Security Considerations

### 1. TLS/SSL Encryption

```bash
# Generate certificates
cd complete
docker compose -f certs.yml run --rm --build gen-certs

# Use TLS in production
docker compose -f compose.yml -f with-tls.yml up -d
```

### 2. Authentication

```python
# Add authentication to MLflow
# In mlflow.Dockerfile
ENV MLFLOW_TRACKING_USERNAME=admin
ENV MLFLOW_TRACKING_PASSWORD=<secure-password>
```

### 3. Network Security

```yaml
# Restrict network access
# In compose.yml
networks:
  fl-network:
    driver: bridge
    internal: true  # No external access

services:
  superlink:
    networks:
      - fl-network
```

### 4. Secrets Management

```bash
# Use Docker secrets
echo "my-secret-key" | docker secret create mlflow_password -

# Reference in compose
services:
  mlflow:
    secrets:
      - mlflow_password
```

---

## Monitoring & Observability

### 1. Prometheus Metrics

```yaml
# Add Prometheus exporter
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
```

### 2. Grafana Dashboards

```yaml
services:
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### 3. Logging

```yaml
# Centralized logging with Loki
services:
  loki:
    image: grafana/loki
    ports:
      - "3100:3100"

  promtail:
    image: grafana/promtail
    volumes:
      - /var/log:/var/log
```

---

## Scaling Strategies

### Horizontal Scaling

```bash
# Scale SuperNodes
docker compose up --scale supernode=20 -d

# Or in Kubernetes
kubectl scale deployment supernode --replicas=50 -n federated-learning
```

### Vertical Scaling

```yaml
# Increase resources
services:
  supernode:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 16G
```

### Auto-scaling (Kubernetes)

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: supernode-hpa
  namespace: federated-learning
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: supernode
  minReplicas: 10
  maxReplicas: 100
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Health Checks

### Docker Compose

```yaml
services:
  superlink:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9093/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Kubernetes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 9093
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 9093
  initialDelaySeconds: 5
  periodSeconds: 5
```

---

## Backup & Recovery

### Backup MLflow Data

```bash
# Backup PostgreSQL
docker exec complete-db-1 pg_dump -U mlflow mlflow > mlflow_backup.sql

# Backup artifacts
docker cp complete-mlflow-1:/mlflow/artifacts ./mlflow_artifacts_backup
```

### Restore

```bash
# Restore database
docker exec -i complete-db-1 psql -U mlflow mlflow < mlflow_backup.sql

# Restore artifacts
docker cp ./mlflow_artifacts_backup complete-mlflow-1:/mlflow/artifacts
```

---

## Performance Tuning

### 1. Database Optimization

```sql
-- PostgreSQL tuning
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
ALTER SYSTEM SET maintenance_work_mem = '1GB';
```

### 2. Network Optimization

```yaml
# Use host network for better performance
services:
  superlink:
    network_mode: "host"
```

### 3. Resource Allocation

```bash
# Monitor resource usage
docker stats

# Adjust based on bottlenecks
# CPU-bound: Increase CPU limits
# Memory-bound: Increase memory limits
# I/O-bound: Use faster storage (SSD)
```

---

## Troubleshooting Production Issues

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed troubleshooting steps.

Quick checks:
```bash
# Check all services
docker compose ps

# View logs
docker compose logs -f

# Check resource usage
docker stats

# Test connectivity
docker exec complete-supernode-1 ping superlink
```

---

## Support

For production deployment support:
- [GitHub Issues](https://github.com/Nikhil172913832/Federated_Learning/issues)
- [GitHub Discussions](https://github.com/Nikhil172913832/Federated_Learning/discussions)
- Email: nikhil172913832@gmail.com
