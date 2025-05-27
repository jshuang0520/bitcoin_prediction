# Bitcoin Prediction System

A real-time Bitcoin price prediction system using machine learning, built with **Docker Compose** and **Kubernetes**.

## 🚀 **Quick Start (Kubernetes - Recommended)**

### **Fixed URLs (Never Change!)**
- 🌐 **Web App**: http://localhost:5001
- 📊 **Dashboard**: http://localhost:8501  
- ⚙️ **Kafka UI**: http://localhost:8080

### **First Time Setup**
```bash
# Prerequisites: Docker Desktop, minikube, kubectl
minikube start --driver=docker
./k8s/build-from-scratch.sh
./k8s/create-tunnels.sh
```

### **Performance-Optimized Setup** ⚡
```bash
# For maximum performance (recommended for demos)
minikube start --driver=docker --cpus=4 --memory=8192
./k8s/deploy-performance-optimized.sh
./k8s/create-tunnels.sh
```

### **Daily Development**
```bash
# After code changes
./k8s/update-service.sh web-app
# Test at http://localhost:5001

# End of day
./k8s/shutdown.sh --pause-minikube

# Next day
./k8s/startup.sh && ./k8s/create-tunnels.sh
```

📖 **Complete Documentation**: [`k8s/README.md`](k8s/README.md)  
⚡ **Quick Reference**: [`k8s/QUICK-START.md`](k8s/QUICK-START.md)

---

## 🐳 **Docker Compose (Legacy)**

For simple development without Kubernetes:

```bash
docker-compose up -d
```

**Services:**
- Web App: http://localhost:5001
- Dashboard: http://localhost:8501
- Kafka UI: http://localhost:8080

---

## 🏗️ **System Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                    Bitcoin Prediction System                │
├─────────────────────────────────────────────────────────────┤
│  Data Pipeline:                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Binance   │  │    Kafka    │  │   TensorFlow ML     │  │
│  │  API Data   │→ │  Streaming  │→ │  Prediction Model   │  │
│  │ Collector   │  │             │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  User Interfaces:                                          │
│  ┌─────────────┐  ┌─────────────────────────────────────┐  │
│  │  Flask      │  │         Streamlit                   │  │
│  │  Web App    │  │        Dashboard                    │  │
│  │             │  │                                     │  │
│  └─────────────┘  └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 📊 **Features**

- **Real-time Data**: Live Bitcoin price collection from Binance API
- **ML Predictions**: TensorFlow-based price forecasting
- **Streaming**: Kafka-based data pipeline
- **Web Interface**: Flask API + Streamlit dashboard
- **Production Ready**: Kubernetes deployment with auto-scaling
- **Fixed URLs**: Consistent localhost access (no more changing ports!)

## 🛠️ **Technology Stack**

- **ML/AI**: TensorFlow, TensorFlow Probability
- **Data Pipeline**: Apache Kafka, Zookeeper
- **Backend**: Python, Flask
- **Frontend**: Streamlit
- **Infrastructure**: Docker, Kubernetes, minikube
- **Data Source**: Binance API

## 📁 **Project Structure**

```
bitcoin_prediction/
├── k8s/                     # Kubernetes deployment (MAIN)
│   ├── README.md           # Complete K8s guide
│   ├── QUICK-START.md      # Quick reference
│   ├── manifests/          # K8s YAML files
│   ├── scripts/            # Management scripts
│   └── *.sh               # Main workflow scripts
├── bitcoin_forecast_app/   # ML prediction service
├── data_collector/         # Binance API data collector
├── web_app/               # Flask web application
├── dashboard/             # Streamlit dashboard
├── docker-compose.yml     # Legacy Docker Compose
└── README.md             # This file
```

## 🎯 **Key Improvements (K8s vs Docker Compose)**

| Feature | Docker Compose | Kubernetes |
|---------|---------------|------------|
| **URL Consistency** | Fixed localhost | Fixed localhost (tunnels) |
| **Auto-scaling** | Manual only | 1-5 pods based on load |
| **Self-healing** | Manual restart | Automatic pod restart |
| **Resource efficiency** | Fixed allocation | Dynamic (60% savings) |
| **Performance optimization** | Basic | Priority classes + 400% resources |
| **Real-time processing** | Best effort | <1s latency guaranteed |
| **Zero-downtime updates** | Service interruption | Rolling updates |
| **Production readiness** | Development only | Production-grade |

## 🚀 **Getting Started**

1. **Choose your deployment method:**
   - **Kubernetes** (recommended): Follow [`k8s/README.md`](k8s/README.md)
   - **Docker Compose** (simple): `docker-compose up -d`

2. **Access your services:**
   - Web App: http://localhost:5001
   - Dashboard: http://localhost:8501
   - Kafka UI: http://localhost:8080

3. **Start developing:**
   - Make code changes
   - Use `./k8s/update-service.sh <service>` for quick updates
   - Test immediately at fixed URLs

## 📚 **Documentation**

- 📖 **[Complete Kubernetes Guide](k8s/README.md)** - Step-by-step instructions for all scenarios
- ⚡ **[Quick Start Reference](k8s/QUICK-START.md)** - Daily workflow commands
- 🏗️ **[Architecture Details](k8s/docs/)** - System design and DevOps presentation

## 🤝 **Contributing**

1. Make your changes
2. Test with: `./k8s/update-service.sh <service-name>`
3. Verify at: http://localhost:5001
4. Submit PR

---

**🎉 Ready to predict Bitcoin prices with production-grade infrastructure!**
