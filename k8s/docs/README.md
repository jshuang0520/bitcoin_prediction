# Bitcoin Prediction System - Kubernetes Deployment

Transform your Docker Compose setup into production-grade Kubernetes with **auto-scaling**, **resource optimization**, and **self-healing**.

## 📁 Project Structure

```
k8s/
├── docs/                    # Documentation
│   ├── README.md           # This file
│   └── DevOps-Presentation.md  # Slide presentation
├── manifests/              # Kubernetes YAML files
│   ├── namespace.yaml      # Namespace definition
│   ├── storage.yaml        # Persistent volumes
│   ├── configmap.yaml      # Configuration
│   ├── zookeeper.yaml      # Zookeeper deployment
│   ├── kafka.yaml          # Kafka deployment
│   ├── kafka-setup.yaml    # Kafka initialization job
│   ├── data-collector.yaml # Data collection service
│   ├── bitcoin-forecast-app.yaml # ML prediction service
│   ├── web-app.yaml        # FastAPI web interface
│   ├── dashboard.yaml      # Streamlit dashboard
│   ├── kafka-ui.yaml       # Kafka management UI
│   └── resource-optimization.yaml # HPA and quotas
└── scripts/                # Management scripts
    ├── start-minikube.sh   # Start Kubernetes cluster
    ├── build-images.sh     # Build Docker images
    ├── deploy.sh           # Deploy all services
    ├── update-service.sh   # Update individual services
    ├── status.sh           # Check system status
    ├── monitor.sh          # Full log monitoring
    ├── demo-monitor.sh     # Clean demo logs
    ├── access.sh           # Service access methods
    └── cleanup.sh          # Stop and cleanup
```

## 🚀 Quick Start (First Time Setup)

### 1. Start Kubernetes Cluster
```bash
# One-command setup (recommended)
./k8s/scripts/start-minikube.sh

# Or manual setup
minikube start --driver=docker --memory=6144 --cpus=4
```

### 2. Build & Deploy Everything
```bash
# Configure Docker for minikube
eval $(minikube docker-env)

# Build all images and deploy (much faster now!)
./k8s/scripts/build-images.sh && ./k8s/scripts/deploy.sh
```

### 3. Access Your Services (Non-blocking)
```bash
# Get all service URLs (non-blocking)
./k8s/scripts/access.sh --all

# Open dashboard in browser (background)
./k8s/scripts/access.sh dashboard --open

# Get specific service URL
./k8s/scripts/access.sh web-app --url
```

## 🔄 Development Workflow

### After Code Changes (Rebuild & Update)
```bash
# For any application changes (bitcoin-forecast-app, web-app, etc.)
eval $(minikube docker-env)  # Only needed once per terminal session
./k8s/scripts/update-service.sh <service-name>

# Examples:
./k8s/scripts/update-service.sh bitcoin-forecast-app  # After ML model changes
./k8s/scripts/update-service.sh web-app              # After web app changes
./k8s/scripts/update-service.sh data-collector       # After data collection changes
./k8s/scripts/update-service.sh dashboard            # After dashboard changes
```

### Quick Rebuild All (After Major Changes)
```bash
eval $(minikube docker-env)
./k8s/scripts/build-images.sh && ./k8s/scripts/deploy.sh
```

## 📊 System Management

### Check System Status
```bash
# Comprehensive status check
./k8s/scripts/status.sh

# Quick pod status
kubectl get pods -n bitcoin-prediction

# Resource usage
kubectl top pods -n bitcoin-prediction
```

### View Logs (Docker-Compose Style)
```bash
# Stream logs from main services (like docker-compose logs -f)
./k8s/scripts/monitor.sh data-collector                    # Stream data-collector logs
./k8s/scripts/monitor.sh bitcoin-forecast-app             # Stream bitcoin-forecast-app logs
./k8s/scripts/monitor.sh web-app                          # Stream web-app logs

# Filter logs with grep (like docker-compose logs -f service | grep pattern)
./k8s/scripts/monitor.sh data-collector --grep "Saved data"        # Filter for data saves
./k8s/scripts/monitor.sh bitcoin-forecast-app --grep "prediction"  # Filter for predictions
./k8s/scripts/monitor.sh bitcoin-forecast-app --grep "error"       # Filter for errors

# Background monitoring (non-blocking terminal)
./k8s/scripts/monitor.sh bitcoin-forecast-app --background
./k8s/scripts/monitor.sh data-collector --background

# Monitor all services in background
./k8s/scripts/monitor.sh --all --background

# Stop all background monitoring
./k8s/scripts/monitor.sh --stop-all
```

### 🎬 Demo Mode (Clean Logs for Presentation)
```bash
# Show only successful predictions (no errors)
./k8s/scripts/demo-monitor.sh predictions

# Show only successful data saves (no errors)
./k8s/scripts/demo-monitor.sh data-saves

# Show both predictions and data saves with color coding
./k8s/scripts/demo-monitor.sh both

# Background demo monitoring
./k8s/scripts/demo-monitor.sh both --background

# Stop demo monitoring
./k8s/scripts/demo-monitor.sh --stop-all
```

### System Control
```bash
# Restart a service
kubectl rollout restart deployment/bitcoin-forecast-app -n bitcoin-prediction

# Scale a service
kubectl scale deployment/web-app --replicas=3 -n bitcoin-prediction

# Shell into a pod
kubectl exec -it deployment/bitcoin-forecast-app -n bitcoin-prediction -- bash
```

## 🌐 Service Access (Non-blocking)

### Get URLs Without Blocking Terminal
```bash
# Get all service URLs
./k8s/scripts/access.sh --all

# Get specific service URL
./k8s/scripts/access.sh dashboard --url
./k8s/scripts/access.sh web-app --url
./k8s/scripts/access.sh kafka-ui --url
```

### Open in Browser (Background)
```bash
# Open services in browser (non-blocking)
./k8s/scripts/access.sh dashboard --open
./k8s/scripts/access.sh web-app --open
./k8s/scripts/access.sh kafka-ui --open
```

### Port Tunneling (Alternative Access)
```bash
# Create localhost tunnels (blocking, but useful for development)
./k8s/scripts/access.sh dashboard --tunnel    # localhost:8501
./k8s/scripts/access.sh web-app --tunnel      # localhost:5000
./k8s/scripts/access.sh kafka-ui --tunnel     # localhost:8080
```

## 🛑 Stopping & Cleanup

### Stop Applications (Keep Infrastructure)
```bash
./k8s/scripts/cleanup.sh --apps-only    # Keep Kafka/Zookeeper running
```

### Stop Everything (Keep Storage)
```bash
./k8s/scripts/cleanup.sh               # Default - keeps data for restart
```

### Complete Cleanup (Delete Everything)
```bash
./k8s/scripts/cleanup.sh --all         # Deletes all data - fresh start
```

### Stop Kubernetes Completely
```bash
./k8s/scripts/cleanup.sh --stop-minikube   # Stops minikube
```

## 🔧 When to Rebuild vs Update

### **Rebuild from Scratch** (Use `./k8s/scripts/deploy.sh`)
- First time setup
- After changing Dockerfile
- After changing Kubernetes YAML files
- After major configuration changes
- When things are broken and you want a fresh start

### **Quick Update** (Use `./k8s/scripts/update-service.sh`)
- After changing Python code
- After changing configuration files
- After changing web app frontend
- For iterative development

### **Just Restart** (Use `kubectl rollout restart`)
- When service is stuck
- After changing ConfigMap values
- For quick troubleshooting

## ⚡ Performance Improvements

### **Faster Deployment (vs Previous Version)**
- ✅ **Parallel waiting**: All services wait in parallel instead of sequentially
- ✅ **Reduced timeouts**: 180s instead of 600s per service
- ✅ **Smarter health checks**: Fixed health check issues causing delays
- ✅ **Background execution**: Non-blocking service access

### **Docker Compose vs Kubernetes Speed Comparison**
| Operation | Docker Compose | Kubernetes (Old) | Kubernetes (New) |
|-----------|---------------|------------------|------------------|
| **Initial Deploy** | ~60s | ~300s | ~120s |
| **Service Update** | ~10s | ~30s | ~15s |
| **Log Monitoring** | Immediate | Immediate | Immediate + Background |
| **Service Access** | Immediate | Blocks terminal | Non-blocking |

## 📈 Key Improvements Over Docker Compose

| Feature | Docker Compose | Kubernetes |
|---------|---------------|------------|
| **Auto-scaling** | Manual only | 1-5 pods based on CPU/memory |
| **Self-healing** | Manual restart | Automatic pod restart |
| **Resource optimization** | Fixed 8GB | Dynamic 2-16GB |
| **Health checks** | Basic | Advanced liveness/readiness |
| **Configuration** | Environment files | Centralized ConfigMaps |
| **Monitoring** | Basic logs | Background + filtered monitoring |
| **Service Access** | Localhost only | Multiple access methods |

## 🚨 Troubleshooting

### Deployment Taking Too Long?
```bash
# Check what's happening
./k8s/scripts/status.sh

# Check specific service
kubectl describe pod -l app=<service-name> -n bitcoin-prediction

# Force restart stuck service
kubectl rollout restart deployment/<service-name> -n bitcoin-prediction
```

### Minikube Issues
```bash
# If minikube won't start
./k8s/scripts/start-minikube.sh        # Automated fix

# Manual troubleshooting
minikube delete && minikube start --driver=docker --memory=6144 --cpus=4
```

### Service Not Working
```bash
# Check service status
./k8s/scripts/status.sh

# Monitor specific service in background
./k8s/scripts/monitor.sh <service-name> --background

# Check logs for errors
./k8s/scripts/monitor.sh <service-name> --grep "error"
```

### Bitcoin Predictions Missing
```bash
# Monitor predictions in real-time (demo mode)
./k8s/scripts/demo-monitor.sh predictions

# Check if bitcoin-forecast-app is working
kubectl get pods -n bitcoin-prediction | grep bitcoin-forecast-app

# Restart if needed
kubectl rollout restart deployment/bitcoin-forecast-app -n bitcoin-prediction
```

## 🎯 Command Reference

| Task | Command |
|------|---------|
| **First time setup** | `./k8s/scripts/start-minikube.sh && eval $(minikube docker-env) && ./k8s/scripts/build-images.sh && ./k8s/scripts/deploy.sh` |
| **Check status** | `./k8s/scripts/status.sh` |
| **Get service URLs** | `./k8s/scripts/access.sh --all` |
| **Open dashboard** | `./k8s/scripts/access.sh dashboard --open` |
| **Stream data-collector** | `./k8s/scripts/monitor.sh data-collector` |
| **Stream bitcoin-forecast-app** | `./k8s/scripts/monitor.sh bitcoin-forecast-app` |
| **Demo mode (clean logs)** | `./k8s/scripts/demo-monitor.sh both` |
| **Background monitoring** | `./k8s/scripts/monitor.sh <service> --background` |
| **Update service** | `./k8s/scripts/update-service.sh <service-name>` |
| **Stop system** | `./k8s/scripts/cleanup.sh` |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                       │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure Layer:                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Zookeeper   │  │   Kafka     │  │  Kafka-UI   │        │
│  │ (Config)    │  │ (Streaming) │  │ (Monitor)   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  Data Pipeline Layer:                                      │
│  ┌─────────────┐  ┌─────────────────────────────────────┐  │
│  │Data         │  │    Bitcoin Forecast App             │  │
│  │Collector    │  │    (TensorFlow ML Model)            │  │
│  │(API→Kafka)  │  │    Auto-scales: 1-3 pods           │  │
│  └─────────────┘  └─────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Frontend Layer:                                           │
│  ┌─────────────┐  ┌─────────────────────────────────────┐  │
│  │  Web App    │  │         Dashboard                   │  │
│  │ (FastAPI)   │  │       (Streamlit)                   │  │
│  │Auto-scales: │  │    Auto-scales: 1-3 pods           │  │
│  │ 1-5 pods    │  │                                     │  │
│  └─────────────┘  └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

All services have:
- **Auto-scaling** based on CPU/memory usage
- **Self-healing** with automatic restarts
- **Resource limits** to prevent system overload
- **Health checks** for reliability
- **Persistent storage** for data retention
- **Background monitoring** capabilities
- **Non-blocking service access**

## 📚 Additional Resources

- 📊 **DevOps Presentation**: `k8s/docs/DevOps-Presentation.md` - Comprehensive slides for interviews
- 🛠️ **Scripts Directory**: `k8s/scripts/` - All management scripts
- 📋 **Manifests Directory**: `k8s/manifests/` - Kubernetes YAML configurations

# Get current URLs (they work consistently)
./k8s/get-urls.sh

# Direct access commands:
minikube service web-app -n bitcoin-prediction
minikube service dashboard -n bitcoin-prediction  
minikube service kafka-ui -n bitcoin-prediction
```

# Your workflow now:
./k8s/get-urls.sh                    # Get current URLs
./k8s/update-service.sh web-app      # Update after code changes  
./k8s/shutdown.sh --pause-minikube   # End of day
./k8s/startup.sh                     # Next day resume