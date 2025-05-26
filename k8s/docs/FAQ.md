# 🤔 Frequently Asked Questions (FAQ)

## 🐳 Docker Images & Containers

### Q1: Why are there always two gcr.io/k8s-minikube/kicbase images with different tags?

**A:** This is normal behavior when using minikube. Here's what happens:

```
REPOSITORY                    TAG       IMAGE ID       CREATED        SIZE
gcr.io/k8s-minikube/kicbase   v0.0.47   631837ba851f   4 days ago     1.76GB
gcr.io/k8s-minikube/kicbase   <none>    6ed579c9292b   4 days ago     1.76GB
```

**Explanation:**
- **v0.0.47**: Current active minikube base image
- **<none>**: Previous version that became "dangling" after update
- This happens when minikube updates its base image

**How to avoid:**
```bash
# Clean up redundant images
./k8s/scripts/cleanup-docker.sh

# Or manually
docker image prune -a -f

# Prevent future duplicates
minikube delete && minikube start  # Instead of just minikube start
```

### Q2: Do we build Kubernetes inside Docker, and then build every Docker image inside that?

**A:** Great question! Here's the actual architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Your MacBook Pro                         │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  🐳 Docker Desktop (Host Docker Daemon)                    │
│  ├─ Your application images:                               │
│  │  ├─ bitcoin-forecast-app:latest                         │
│  │  ├─ data-collector:latest                               │
│  │  ├─ web-app:latest                                      │
│  │  └─ dashboard:latest                                    │
│  │                                                         │
│  └─ 🎯 Minikube Container (Kubernetes-in-Docker)          │
│      ├─ Contains: Kubernetes cluster                       │
│      ├─ Shares Docker daemon with host                     │
│      └─ Runs your application containers                   │
└─────────────────────────────────────────────────────────────┘
```

**Key Points:**
1. **Minikube runs as a container** inside Docker Desktop
2. **Images are built on the host** Docker daemon
3. **`eval $(minikube docker-env)`** makes your shell use minikube's Docker daemon
4. **Images are shared** between host and minikube

**Why you only see kicbase:**
- When you run `docker images` on host, you see host images
- When you run `eval $(minikube docker-env)` then `docker images`, you see minikube's images
- `kicbase` is the base image for the minikube container itself

### Q3: How do Docker images and containers relate?

**A:** Think of it like classes and objects in programming:

```
📦 DOCKER IMAGE (Class/Blueprint)
├─ Read-only template
├─ Contains: OS + dependencies + application code
├─ Stored in layers for efficiency
└─ Example: bitcoin-forecast-app:latest

🏃 DOCKER CONTAINER (Object/Instance)
├─ Running instance of an image
├─ Has writable layer on top
├─ Isolated process with own filesystem
└─ Example: bitcoin-forecast-app-7cd596c84-rgcph
```

**In Kubernetes:**
- **Pod** = Wrapper around one or more containers
- **Deployment** = Manages multiple pods (replicas)
- **Service** = Network access to pods

## ☸️ Kubernetes Resource Allocation

### Q4: Does minikube only use 6GB memory and 4 CPUs for everything?

**A:** Yes, but here's how it's distributed:

```bash
minikube start --driver=docker --memory=6144 --cpus=4
```

**Resource Breakdown:**

```
┌─────────────────────────────────────────────────────────────┐
│  🎯 Minikube VM: 6GB RAM, 4 CPUs (Total Allocation)        │
├─────────────────────────────────────────────────────────────┤
│  ☸️ Kubernetes System (Infrastructure): ~1GB RAM, 0.5 CPU  │
│  ├─ kube-apiserver:        ~200MB RAM, 0.1 CPU            │
│  ├─ etcd:                  ~100MB RAM, 0.1 CPU            │
│  ├─ kube-scheduler:        ~50MB RAM,  0.05 CPU           │
│  ├─ kube-controller:       ~100MB RAM, 0.1 CPU            │
│  ├─ kube-proxy:            ~50MB RAM,  0.05 CPU           │
│  ├─ coredns:               ~100MB RAM, 0.05 CPU           │
│  └─ storage-provisioner:   ~50MB RAM,  0.05 CPU           │
├─────────────────────────────────────────────────────────────┤
│  🚀 Our Applications: ~5GB RAM, 3.5 CPUs (Available)      │
│  ├─ Zookeeper:            200MB RAM,  0.2 CPU             │
│  ├─ Kafka:                512MB RAM,  0.5 CPU             │
│  ├─ Data-collector:       256MB RAM,  0.3 CPU             │
│  ├─ Bitcoin-forecast:     1GB RAM,    1.0 CPU             │
│  ├─ Web-app:              256MB RAM,  0.3 CPU             │
│  ├─ Dashboard:            512MB RAM,  0.5 CPU             │
│  └─ Kafka-UI:             256MB RAM,  0.2 CPU             │
└─────────────────────────────────────────────────────────────┘
```

**Dynamic Scaling:**
- **Minimum**: Apps use ~2GB RAM, 1.5 CPUs during low load
- **Maximum**: Apps can scale up to 5GB RAM, 3.5 CPUs during high load
- **Auto-scaling**: Triggers at 70-80% CPU/memory usage

### Q5: How does auto-scaling work?

**A:** Kubernetes Horizontal Pod Autoscaler (HPA) monitors resource usage:

```yaml
# Example: Web-app auto-scaling
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: web-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: web-app
  minReplicas: 1          # Minimum pods
  maxReplicas: 5          # Maximum pods
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60    # Scale up at 60% CPU
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 70    # Scale up at 70% memory
```

**Scaling Behavior:**
1. **Low Load**: 1 pod using 256MB RAM, 0.1 CPU
2. **Medium Load**: 2-3 pods using 512-768MB RAM, 0.2-0.3 CPU
3. **High Load**: 4-5 pods using 1-1.25GB RAM, 0.4-0.5 CPU

## 🎬 Demo & Monitoring

### Q6: How do I show only clean logs during demo?

**A:** Use the demo monitoring script:

```bash
# Show only successful predictions (no errors)
./k8s/scripts/demo-monitor.sh predictions

# Show only successful data saves (no errors)  
./k8s/scripts/demo-monitor.sh data-saves

# Show both with color coding
./k8s/scripts/demo-monitor.sh both

# Background monitoring for demo
./k8s/scripts/demo-monitor.sh both --background
```

**What it filters:**
- ✅ **Predictions**: Only "Made prediction for timestamp" messages
- ✅ **Data Saves**: Only "Saved data to" messages
- ❌ **Errors**: Filtered out (AttributeError, connection issues, etc.)
- ❌ **Debug**: Filtered out (verbose TensorFlow logs, etc.)

### Q7: How do I compare Docker Compose vs Kubernetes for interviews?

**A:** Use the presentation slides:

```bash
# View comprehensive presentation
cat k8s/docs/DevOps-Presentation.md

# Key talking points:
# 1. Resource efficiency (60% memory savings)
# 2. Auto-scaling (1-5 replicas vs fixed)
# 3. Self-healing (automatic vs manual)
# 4. Zero-downtime updates
# 5. Production-grade monitoring
```

**Demo Script for Interviews:**
```bash
# 1. Show system status
./k8s/scripts/status.sh

# 2. Show real-time data collection
./k8s/scripts/demo-monitor.sh data-saves

# 3. Show ML predictions
./k8s/scripts/demo-monitor.sh predictions

# 4. Demonstrate auto-scaling
kubectl scale deployment/web-app --replicas=3 -n bitcoin-prediction
kubectl get hpa -n bitcoin-prediction -w

# 5. Show self-healing
kubectl delete pod -l app=bitcoin-forecast-app -n bitcoin-prediction
kubectl get pods -n bitcoin-prediction -w
```

## 🏗️ Project Organization

### Q8: Should we reorganize the k8s directory?

**A:** ✅ **Already done!** The k8s directory is now properly organized:

```
k8s/
├── docs/                    # 📚 Documentation
│   ├── README.md           # Main documentation
│   ├── DevOps-Presentation.md  # Interview slides
│   └── FAQ.md              # This file
├── manifests/              # 📋 Kubernetes YAML files
│   ├── namespace.yaml      # Namespace definition
│   ├── storage.yaml        # Persistent volumes
│   ├── configmap.yaml      # Configuration
│   ├── *.yaml              # Service deployments
│   └── resource-optimization.yaml # HPA and quotas
└── scripts/                # 🛠️ Management scripts
    ├── start-minikube.sh   # Start cluster
    ├── deploy.sh           # Deploy all services
    ├── demo-monitor.sh     # Clean demo logs
    ├── cleanup-docker.sh   # Clean redundant images
    └── *.sh                # Other management scripts
```

**Benefits:**
- ✅ **Clear separation** of concerns
- ✅ **Easy navigation** for developers
- ✅ **Professional structure** for interviews
- ✅ **Scalable organization** for future features

## 🚀 Getting Started

### Q9: How do I follow the README guide?

**A:** The README has been updated with the new structure:

```bash
# 1. First time setup
./k8s/scripts/start-minikube.sh
eval $(minikube docker-env)
./k8s/scripts/build-images.sh && ./k8s/scripts/deploy.sh

# 2. Check status
./k8s/scripts/status.sh

# 3. Demo monitoring (clean logs)
./k8s/scripts/demo-monitor.sh both

# 4. Access services
./k8s/scripts/access.sh --all

# 5. Update after code changes
./k8s/scripts/update-service.sh bitcoin-forecast-app
```

### Q10: What's the difference between the monitoring scripts?

**A:** Two different monitoring approaches:

```bash
# 1. Full monitoring (shows everything including errors)
./k8s/scripts/monitor.sh bitcoin-forecast-app
./k8s/scripts/monitor.sh data-collector --grep "Saved data"

# 2. Demo monitoring (clean logs only, no errors)
./k8s/scripts/demo-monitor.sh predictions
./k8s/scripts/demo-monitor.sh data-saves
./k8s/scripts/demo-monitor.sh both
```

**Use cases:**
- **Development**: Use `monitor.sh` to see all logs including errors
- **Demo/Interview**: Use `demo-monitor.sh` to show clean, professional output
- **Debugging**: Use `monitor.sh` with `--grep "error"` to find issues

## 🔧 Troubleshooting

### Q11: How do I clean up redundant Docker images?

**A:** Use the cleanup script:

```bash
# Interactive cleanup
./k8s/scripts/cleanup-docker.sh

# Manual cleanup
docker image prune -a -f    # Remove unused images
docker builder prune -f     # Remove build cache
```

### Q12: How do I prevent kicbase duplication?

**A:** Best practices:

```bash
# Method 1: Clean restart
minikube delete
minikube start --driver=docker --memory=6144 --cpus=4

# Method 2: Force restart (reuses existing)
minikube start --force

# Method 3: Regular cleanup
./k8s/scripts/cleanup-docker.sh
```

## 📊 Performance & Resources

### Q13: How does resource allocation compare to Docker Compose?

**A:** Detailed comparison:

| Aspect | Docker Compose | Kubernetes |
|--------|---------------|------------|
| **Memory (Idle)** | 8GB fixed | 2GB dynamic |
| **Memory (Peak)** | 8GB fixed | 5GB dynamic |
| **CPU (Idle)** | 6 cores fixed | 1 core dynamic |
| **CPU (Peak)** | 6 cores fixed | 4 cores dynamic |
| **Scaling** | Manual | Automatic (HPA) |
| **Recovery** | Manual restart | Auto-healing |
| **Updates** | Downtime | Zero-downtime |

**Cost Savings:**
- **Development**: 60-75% resource savings during idle
- **Production**: 30-60% savings through auto-scaling
- **Operational**: Reduced manual intervention

## 🎯 Interview Preparation

### Q14: What should I highlight in a DevOps interview?

**A:** Key talking points:

1. **Infrastructure as Code**: YAML manifests in version control
2. **Auto-scaling**: HPA based on CPU/memory metrics
3. **Self-healing**: Automatic pod restart and rescheduling
4. **Zero-downtime deployments**: Rolling updates
5. **Resource optimization**: Dynamic allocation vs fixed
6. **Monitoring & Observability**: Structured logging and metrics
7. **Production readiness**: Health checks, resource limits, quotas

**Demo Flow:**
```bash
# Show current state
./k8s/scripts/status.sh

# Show real-time processing
./k8s/scripts/demo-monitor.sh both

# Demonstrate scaling
kubectl get hpa -n bitcoin-prediction

# Show self-healing
kubectl delete pod -l app=bitcoin-forecast-app -n bitcoin-prediction
kubectl get pods -n bitcoin-prediction -w
```

---

## 📚 Additional Resources

- 📖 **Main Documentation**: `k8s/docs/README.md`
- 🎯 **Interview Slides**: `k8s/docs/DevOps-Presentation.md`
- 🛠️ **All Scripts**: `k8s/scripts/`
- 📋 **Kubernetes Manifests**: `k8s/manifests/`

**Quick Commands:**
```bash
# System status
./k8s/scripts/status.sh

# Demo mode
./k8s/scripts/demo-monitor.sh both

# Clean Docker
./k8s/scripts/cleanup-docker.sh

# Access services
./k8s/scripts/access.sh --all
``` 