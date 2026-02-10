# Kubernetes Deployment Guide

## Prerequisites

- Kubernetes cluster (GKE, EKS, or local with minikube)
- kubectl configured
- Helm 3.x installed
- Docker for building images

## Quick Deploy

### 1. Build Docker Image

```bash
cd complete/
docker build -t fl-platform:latest .
```

### 2. Deploy with Helm

```bash
helm install fl-platform helm/fl-platform \
  --namespace fl-system \
  --create-namespace
```

### 3. Verify Deployment

```bash
kubectl get pods -n fl-system
kubectl get svc -n fl-system
```

## Configuration

### Custom Values

Create `custom-values.yaml`:

```yaml
replicaCount:
  client: 10

config:
  numRounds: 50
  learningRate: 0.001
  
privacy:
  enabled: true
  epsilon: 3.0
```

Deploy with custom values:

```bash
helm upgrade fl-platform helm/fl-platform \
  -f custom-values.yaml \
  --namespace fl-system
```

## Monitoring

### Access Grafana

```bash
kubectl port-forward svc/grafana 3000:3000 -n fl-system
```

Visit http://localhost:3000 (admin/admin)

### Access MLflow

```bash
kubectl port-forward svc/mlflow 5000:5000 -n fl-system
```

Visit http://localhost:5000

### Access Prometheus

```bash
kubectl port-forward svc/prometheus 9090:9090 -n fl-system
```

Visit http://localhost:9090

## Scaling

### Manual Scaling

```bash
kubectl scale deployment fl-client --replicas=20 -n fl-system
```

### Auto-scaling

HPA is configured by default:
- Min replicas: 2
- Max replicas: 20
- Target CPU: 70%
- Target Memory: 80%

## Troubleshooting

### Check Logs

```bash
# Server logs
kubectl logs -f deployment/fl-server -n fl-system

# Client logs
kubectl logs -f deployment/fl-client -n fl-system

# MLflow logs
kubectl logs -f statefulset/mlflow -n fl-system
```

### Common Issues

**Pods not starting:**
```bash
kubectl describe pod <pod-name> -n fl-system
```

**Out of resources:**
```bash
kubectl top nodes
kubectl top pods -n fl-system
```

## Cleanup

```bash
helm uninstall fl-platform -n fl-system
kubectl delete namespace fl-system
```
