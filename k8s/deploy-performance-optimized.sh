#!/bin/bash

# Bitcoin Prediction System - Deploy Performance Optimized Version
# This script deploys the enhanced system with significant performance improvements

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_header() {
    echo ""
    echo -e "${CYAN}=== $1 ===${NC}"
    echo ""
}

# Check if minikube is running
if ! minikube status &> /dev/null; then
    print_error "Minikube is not running. Run: minikube start --driver=docker"
    exit 1
fi

print_header "🚀 Deploying Performance-Optimized Bitcoin Prediction System"

print_info "This deployment includes:"
echo -e "  ${GREEN}✅ +400% CPU/Memory for Data-Collector${NC}"
echo -e "  ${GREEN}✅ +100-300% CPU/Memory for ML Processing${NC}"
echo -e "  ${GREEN}✅ Priority-based resource scheduling${NC}"
echo -e "  ${GREEN}✅ Enhanced auto-scaling with faster response${NC}"
echo -e "  ${GREEN}✅ Real-time processing optimizations${NC}"
echo ""

# Apply performance optimizations first
print_header "🎯 Applying Performance Optimizations"

print_info "Applying priority classes and enhanced auto-scaling..."
kubectl apply -f k8s/manifests/performance-optimization.yaml
print_status "Performance optimizations applied"

# Apply namespace and basic resources
print_info "Setting up namespace and storage..."
kubectl apply -f k8s/manifests/namespace.yaml
kubectl apply -f k8s/manifests/storage.yaml
kubectl apply -f k8s/manifests/configmap.yaml
print_status "Basic resources configured"

# Deploy infrastructure services with enhanced resources
print_header "🏗️ Deploying Enhanced Infrastructure"

print_info "Deploying ZooKeeper with high priority..."
kubectl apply -f k8s/manifests/zookeeper.yaml
print_status "ZooKeeper deployed"

print_info "Deploying Kafka with critical priority and 4x resources..."
kubectl apply -f k8s/manifests/kafka.yaml
print_status "Kafka deployed"

# Wait for infrastructure to be ready
print_info "Waiting for infrastructure services to be ready..."
kubectl wait --for=condition=available --timeout=120s deployment/zookeeper -n bitcoin-prediction
kubectl wait --for=condition=available --timeout=180s deployment/kafka -n bitcoin-prediction
print_status "Infrastructure services ready"

# Setup Kafka topics
print_info "Setting up Kafka topics..."
kubectl apply -f k8s/manifests/kafka-setup.yaml
print_status "Kafka setup job created"

# Deploy application services with enhanced resources
print_header "⚡ Deploying Enhanced Application Services"

print_info "Deploying Data-Collector with 5x CPU and 4x memory..."
kubectl apply -f k8s/manifests/data-collector.yaml
print_status "Data-Collector deployed"

print_info "Deploying Bitcoin-Forecast-App with critical priority and 2x CPU, 4x memory..."
kubectl apply -f k8s/manifests/bitcoin-forecast-app.yaml
print_status "Bitcoin-Forecast-App deployed"

print_info "Deploying Web-App..."
kubectl apply -f k8s/manifests/web-app.yaml
print_status "Web-App deployed"

print_info "Deploying Dashboard..."
kubectl apply -f k8s/manifests/dashboard.yaml
print_status "Dashboard deployed"

print_info "Deploying Kafka-UI..."
kubectl apply -f k8s/manifests/kafka-ui.yaml
print_status "Kafka-UI deployed"

# Wait for all services to be ready
print_header "⏳ Waiting for All Services to be Ready"

services=("data-collector" "bitcoin-forecast-app" "web-app" "dashboard" "kafka-ui")

for service in "${services[@]}"; do
    print_info "Waiting for $service to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/$service -n bitcoin-prediction
    print_status "$service is ready"
done

print_header "📊 Performance Optimization Summary"

echo -e "${CYAN}🔧 Resource Enhancements Applied:${NC}"
echo -e "  ${GREEN}✅ Data-Collector: 100m→500m CPU (+400%), 128Mi→512Mi RAM (+300%)${NC}"
echo -e "  ${GREEN}✅ Bitcoin-Forecast-App: 1000m→2000m CPU (+100%), 1Gi→4Gi RAM (+300%)${NC}"
echo -e "  ${GREEN}✅ Kafka: 250m→1000m CPU (+300%), 512Mi→2Gi RAM (+300%)${NC}"
echo -e "  ${GREEN}✅ ZooKeeper: 100m→250m CPU (+150%), 256Mi→512Mi RAM (+100%)${NC}"

echo ""
echo -e "${CYAN}🎯 Priority Classes Configured:${NC}"
echo -e "  ${RED}🔴 Critical (1000): Bitcoin-Forecast-App, Kafka${NC}"
echo -e "  ${YELLOW}🟡 High (500): Data-Collector, ZooKeeper${NC}"
echo -e "  ${GREEN}🟢 Normal (100): Web-App, Dashboard, Kafka-UI${NC}"

echo ""
echo -e "${CYAN}🔄 Enhanced Auto-Scaling:${NC}"
echo -e "  ${GREEN}✅ Data-Collector: 1-2 replicas, 30s scale-up${NC}"
echo -e "  ${GREEN}✅ Bitcoin-Forecast-App: 1-3 replicas, 60s scale-up${NC}"
echo -e "  ${GREEN}✅ Web-App: 1-5 replicas, load-based scaling${NC}"
echo -e "  ${GREEN}✅ Dashboard: 1-3 replicas, responsive scaling${NC}"

echo ""
echo -e "${CYAN}⚡ Performance Optimizations:${NC}"
echo -e "  ${GREEN}✅ TensorFlow multi-threading (4 cores)${NC}"
echo -e "  ${GREEN}✅ Kafka high-throughput configuration${NC}"
echo -e "  ${GREEN}✅ Python optimization flags${NC}"
echo -e "  ${GREEN}✅ Enhanced health checks (faster detection)${NC}"
echo -e "  ${GREEN}✅ Node affinity for optimal placement${NC}"

print_header "🌐 Service Access"

print_info "Creating fixed URL tunnels..."
if [ -f "k8s/create-tunnels.sh" ]; then
    ./k8s/create-tunnels.sh
else
    print_warning "Tunnel script not found. Services available via minikube service commands."
fi

print_header "📈 Performance Monitoring"

print_info "Performance monitoring tools available:"
echo ""
echo "  # Real-time performance monitoring:"
echo "  ./k8s/scripts/performance-monitor.sh"
echo ""
echo "  # Watch resource usage:"
echo "  watch kubectl top pods -n bitcoin-prediction"
echo ""
echo "  # Monitor auto-scaling:"
echo "  watch kubectl get hpa -n bitcoin-prediction"
echo ""
echo "  # Check system status:"
echo "  ./k8s/status.sh"

print_header "🎉 Performance-Optimized Deployment Complete!"

print_status "All services deployed with enhanced performance configurations"
print_info "Expected improvements:"
echo -e "  ${GREEN}✅ Sub-second data collection latency${NC}"
echo -e "  ${GREEN}✅ <1s prediction cycle time${NC}"
echo -e "  ${GREEN}✅ Real-time dashboard updates${NC}"
echo -e "  ${GREEN}✅ High-throughput Kafka streaming${NC}"
echo -e "  ${GREEN}✅ Automatic scaling based on load${NC}"

echo ""
print_info "🔍 Monitor performance with: ./k8s/scripts/performance-monitor.sh"
print_info "🌐 Access services at: http://localhost:5001, http://localhost:8501, http://localhost:8080" 