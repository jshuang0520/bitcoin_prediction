#!/bin/bash

# Bitcoin Prediction System - Build from Scratch
# This script builds everything from zero and deploys to Kubernetes

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

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_header() {
    echo ""
    echo -e "${BLUE}=== $1 ===${NC}"
    echo ""
}

print_header "🏗️ Building Bitcoin Prediction System from Scratch"

# Check if Docker is running
if ! docker info &> /dev/null; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed. Please install kubectl and try again."
    exit 1
fi

# Check if minikube is available
if ! command -v minikube &> /dev/null; then
    print_error "minikube is not installed. Please install minikube and try again."
    exit 1
fi

# Start minikube if not running
print_info "Starting minikube..."
if ! minikube status &> /dev/null; then
    minikube start --driver=docker
    print_status "Minikube started successfully"
else
    print_status "Minikube is already running"
fi

# Update context to ensure we're pointing to the right cluster
minikube update-context
print_status "Kubectl context updated"

# Configure Docker environment for minikube
print_info "Configuring Docker environment for minikube..."
eval $(minikube docker-env)
print_status "Docker environment configured"

# Build all Docker images
print_header "🔨 Building Docker Images"

# Navigate to project root
cd "$(dirname "$0")/.."

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
docker build -f web_app/Dockerfile -t web-app:latest ./web_app
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

# Deploy application services in parallel for faster deployment
print_info "Deploying application services..."
kubectl apply -f k8s/manifests/data-collector.yaml
kubectl apply -f k8s/manifests/bitcoin-forecast-app.yaml
kubectl apply -f k8s/manifests/web-app.yaml
kubectl apply -f k8s/manifests/dashboard.yaml
kubectl apply -f k8s/manifests/kafka-ui.yaml
print_status "Application services deployed"

# Apply resource optimization
print_info "Applying resource optimization..."
kubectl apply -f k8s/manifests/resource-optimization.yaml
print_status "Resource optimization applied"

# Wait for all services to be ready (parallel)
print_info "Waiting for all services to be ready..."
{
    kubectl wait --for=condition=available --timeout=300s deployment/data-collector -n bitcoin-prediction &
    kubectl wait --for=condition=available --timeout=300s deployment/bitcoin-forecast-app -n bitcoin-prediction &
    kubectl wait --for=condition=available --timeout=300s deployment/web-app -n bitcoin-prediction &
    kubectl wait --for=condition=available --timeout=300s deployment/dashboard -n bitcoin-prediction &
    kubectl wait --for=condition=available --timeout=300s deployment/kafka-ui -n bitcoin-prediction &
    wait
} && print_status "All services are ready!" || print_warning "Some services may still be starting up"

# Show deployment status
print_header "📊 Deployment Status"
kubectl get pods -n bitcoin-prediction

# Get minikube IP for fixed URLs
MINIKUBE_IP=$(minikube ip)

# Show access information
print_header "🌐 Service Access Information"
echo "Your Bitcoin Prediction System is now running!"
echo ""
echo "📱 Fixed Service URLs (always the same):"
echo "  Web App:    http://$MINIKUBE_IP:30001"
echo "  Dashboard:  http://$MINIKUBE_IP:30002"
echo "  Kafka UI:   http://$MINIKUBE_IP:30003"
echo ""
echo "📋 Quick Management Commands:"
echo "  Status:     ./k8s/status.sh"
echo "  Logs:       ./k8s/logs.sh <service-name>"
echo "  Update:     ./k8s/update-service.sh <service-name>"
echo "  Shutdown:   ./k8s/shutdown.sh"
echo ""

# Test web-app accessibility
print_info "Testing web-app accessibility..."
if curl -s --connect-timeout 5 http://$MINIKUBE_IP:30001 > /dev/null; then
    print_status "Web-app is accessible at http://$MINIKUBE_IP:30001"
else
    print_warning "Web-app may still be starting up. Please wait a moment and try accessing http://$MINIKUBE_IP:30001"
fi

print_header "🎉 Build and Deployment Complete!"
echo "Your Bitcoin prediction system is ready for use!"
echo ""
echo "Next steps:"
echo "1. Open http://$MINIKUBE_IP:30001 in your browser"
echo "2. Check the dashboard at http://$MINIKUBE_IP:30002"
echo "3. Monitor system with ./k8s/status.sh"
echo ""

print_status "Build from scratch completed successfully!" 