# 🚀 Bitcoin Prediction System - Quick Start

## **FIXED URLS (Never Change!)**
- 🌐 **Web App**: http://localhost:5001
- 📊 **Dashboard**: http://localhost:8501  
- ⚙️ **Kafka UI**: http://localhost:8080

---

## **5 Main Scenarios**

### 1️⃣ **First Time / From Scratch (Standard)**
```bash
./k8s/build-from-scratch.sh     # Auto-handles minikube
./k8s/create-tunnels.sh
```
**Result**: All services built and running with fixed URLs

### 1️⃣🚀 **First Time / Performance Optimized**
```bash
./k8s/build-performance-optimized.sh  # Auto-handles minikube + enhanced config
./k8s/create-tunnels.sh
```
**Result**: Maximum performance with 3x faster processing + auto-scaling

### 2️⃣ **Update After Code Changes**  
```bash
./k8s/update-service.sh web-app
# Test immediately at http://localhost:5001
```
**Result**: Only changed service rebuilt (1-2 min vs 5-8 min)

### 3️⃣ **End of Day / Save Resources**
```bash
./k8s/create-tunnels.sh --stop
./k8s/shutdown.sh --pause-minikube
```
**Result**: All data preserved, maximum resources saved

### 4️⃣ **Next Day / Resume Work**
```bash
./k8s/startup.sh
./k8s/create-tunnels.sh
```
**Result**: Everything restored, same fixed URLs

---

## **Performance Comparison**

| Feature | Standard Build | Performance Optimized |
|---------|---------------|----------------------|
| **Data Collection** | ~2-3s delays | <1s consistent |
| **ML Predictions** | ~5-10s processing | ~1-2s processing |
| **Resource Usage** | ~40% efficiency | ~75% efficiency |
| **Auto-scaling** | Manual only | 1-5 pods automatic |
| **Minikube Config** | 6GB RAM, 4 CPUs | 8GB RAM, 6 CPUs |

---

## **Daily Workflow Example**

```bash
# Morning (30 seconds)
./k8s/startup.sh && ./k8s/create-tunnels.sh

# During development (1-2 min per change)
# Edit code, then:
./k8s/update-service.sh web-app
# Test at http://localhost:5001

# Monitor performance (optional)
./k8s/monitor-performance.sh

# End of day (30 seconds)
./k8s/shutdown.sh --pause-minikube
```

---

## **Problem Solved: Fixed URLs**

❌ **Before**: Ports change every restart (http://127.0.0.1:62728 → http://127.0.0.1:65432)  
✅ **After**: Always the same (http://localhost:5001)

❌ **Before**: Terminal blocked by minikube service tunnels  
✅ **After**: Background tunnels, terminal stays free

❌ **Before**: Need to run `minikube service` every time  
✅ **After**: Just bookmark http://localhost:5001

---

## **Status & Management**

```bash
./k8s/status.sh                  # Check everything
./k8s/logs.sh web-app -f         # Follow logs  
./k8s/create-tunnels.sh --stop   # Stop tunnels
./k8s/monitor-performance.sh     # Performance metrics
```

---

## **Prerequisites (One-time Setup)**

1. **Install Docker Desktop** (must be running)
2. **Install minikube**: `brew install minikube`
3. **Install kubectl**: `brew install kubectl`

---

**🎯 Total Setup Time: 5-8 minutes first time, then 1-2 minutes for updates** 