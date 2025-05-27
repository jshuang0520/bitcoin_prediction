#!/bin/bash

# Bitcoin Prediction System - Simple Deployment
# This script builds and deploys with proper resource management

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_header() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
    echo ""
}

print_header "🏗️ Building Bitcoin Prediction System"

# Configure Docker environment for minikube
print_info "Configuring Docker environment for minikube..."
eval $(minikube docker-env)
print_status "Docker environment configured"

# Build all Docker images
print_header "🔨 Building Docker Images"

# Build data-collector
print_info "Building data-collector image..."
docker build -f data_collector/Dockerfile -t data-collector:latest .
print_status "data-collector image built"

# Build bitcoin-forecast-app
print_info "Building bitcoin-forecast-app image..."
docker build -f bitcoin_forecast_app/Dockerfile -t bitcoin-forecast-app:latest .
print_status "bitcoin-forecast-app image built"

# Build web-app
print_info "Building web-app image..."
docker build -f web_app/Dockerfile -t web-app:latest .
print_status "web-app image built"

# Build dashboard
print_info "Building dashboard image..."
docker build -f dashboard/Dockerfile -t dashboard:latest .
print_status "dashboard image built"

print_status "All Docker images built successfully"

# Deploy to Kubernetes
print_header "🚀 Deploying to Kubernetes"

# Create namespace
print_info "Creating namespace..."
kubectl apply -f k8s/manifests/namespace.yaml
print_status "Namespace created"

# Create storage
print_info "Creating storage resources..."
kubectl apply -f k8s/manifests/storage.yaml
print_status "Storage resources created"

# Create configmap
print_info "Creating configuration..."
kubectl apply -f k8s/manifests/configmap.yaml
print_status "Configuration created"

# Apply performance optimization first (for priority classes)
print_info "Applying performance optimization..."
kubectl apply -f k8s/manifests/performance-optimization.yaml
print_status "Performance optimization applied"

# Deploy Zookeeper first
print_info "Deploying Zookeeper..."
kubectl apply -f k8s/manifests/zookeeper.yaml
kubectl wait --for=condition=available --timeout=120s deployment/zookeeper -n bitcoin-prediction
print_status "Zookeeper deployed and ready"

# Deploy Kafka
print_info "Deploying Kafka..."
kubectl apply -f k8s/manifests/kafka.yaml
kubectl wait --for=condition=available --timeout=180s deployment/kafka -n bitcoin-prediction
print_status "Kafka deployed and ready"

# Run Kafka setup job
print_info "Setting up Kafka topics..."
kubectl apply -f k8s/manifests/kafka-setup.yaml
kubectl wait --for=condition=complete --timeout=120s job/kafka-setup -n bitcoin-prediction
print_status "Kafka topics created"

# Deploy application services
print_info "Deploying application services..."
kubectl apply -f k8s/manifests/data-collector.yaml
kubectl apply -f k8s/manifests/bitcoin-forecast-app.yaml
kubectl apply -f k8s/manifests/web-app.yaml
kubectl apply -f k8s/manifests/dashboard.yaml
kubectl apply -f k8s/manifests/kafka-ui.yaml
print_status "Application services deployed"

# Wait for services to be ready
print_info "Waiting for services to be ready..."
sleep 30

# Show deployment status
print_header "📊 Deployment Status"
kubectl get pods -n bitcoin-prediction

# Get minikube IP for fixed URLs
MINIKUBE_IP=$(minikube ip)

# Show access information
print_header "🌐 Service Access Information"
echo "Your Bitcoin Prediction System is now running!"
echo ""
echo "📱 Fixed Service URLs:"
echo "  Web App:    http://$MINIKUBE_IP:30001"
echo "  Dashboard:  http://$MINIKUBE_IP:30002"
echo "  Kafka UI:   http://$MINIKUBE_IP:30003"
echo ""

print_header "🎉 Deployment Complete!"
echo "Your Bitcoin prediction system is ready for use!"
echo ""
echo "Next steps:"
echo "1. Open http://$MINIKUBE_IP:30001 in your browser"
echo "2. Check the dashboard at http://$MINIKUBE_IP:30002"
echo ""

print_status "Simple deployment completed successfully!" 