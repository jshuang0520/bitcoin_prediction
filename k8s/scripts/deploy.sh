#!/bin/bash

# Bitcoin Prediction System - Kubernetes Deployment Script
# This script deploys the complete Bitcoin prediction system to Kubernetes

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

echo "🚀 Starting Bitcoin Prediction System deployment..."

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

# Check if we can connect to Kubernetes cluster
print_info "Waiting for Kubernetes cluster to be ready..."
if ! kubectl cluster-info &> /dev/null; then
    print_error "Cannot connect to Kubernetes cluster"
    print_info "Make sure minikube is running: ./k8s/start-minikube.sh"
    exit 1
fi
print_status "Connected to Kubernetes cluster"

# Deploy base resources
print_info "Creating namespace and base resources..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/storage.yaml
kubectl apply -f k8s/configmap.yaml
print_status "Base resources created"

# Deploy infrastructure services
print_info "Deploying infrastructure services..."
kubectl apply -f k8s/zookeeper.yaml
print_status "Zookeeper deployed"

print_info "Waiting for Zookeeper to be ready..."
kubectl wait --for=condition=available --timeout=120s deployment/zookeeper -n bitcoin-prediction
kubectl apply -f k8s/kafka.yaml
print_status "Kafka deployed"

print_info "Waiting for Kafka to be ready..."
kubectl wait --for=condition=available --timeout=120s deployment/kafka -n bitcoin-prediction

# Setup Kafka topics
print_info "Setting up Kafka topics..."
kubectl apply -f k8s/kafka-setup.yaml
print_status "Kafka setup job started"

print_info "Waiting for Kafka setup to complete..."
kubectl wait --for=condition=complete --timeout=120s job/kafka-setup -n bitcoin-prediction

# Deploy application services
print_info "Deploying application services..."
kubectl apply -f k8s/data-collector.yaml
kubectl apply -f k8s/bitcoin-forecast-app.yaml
kubectl apply -f k8s/web-app.yaml
kubectl apply -f k8s/dashboard.yaml
kubectl apply -f k8s/kafka-ui.yaml
print_status "Application services deployed"

# Apply resource optimization
print_info "Applying resource optimization..."
kubectl apply -f k8s/resource-optimization.yaml
print_status "Resource optimization applied"

# Wait for all services in parallel (much faster!)
print_info "Waiting for all services to be ready (parallel)..."
{
    kubectl wait --for=condition=available --timeout=180s deployment/data-collector -n bitcoin-prediction &
    kubectl wait --for=condition=available --timeout=180s deployment/bitcoin-forecast-app -n bitcoin-prediction &
    kubectl wait --for=condition=available --timeout=180s deployment/web-app -n bitcoin-prediction &
    kubectl wait --for=condition=available --timeout=180s deployment/dashboard -n bitcoin-prediction &
    kubectl wait --for=condition=available --timeout=180s deployment/kafka-ui -n bitcoin-prediction &
    wait
} && print_status "All services are ready!" || print_warning "Some services may still be starting up"

# Show final status
echo ""
print_status "🎉 Bitcoin Prediction System deployed successfully!"
echo ""
print_info "📊 System Status:"
kubectl get pods -n bitcoin-prediction

echo ""
print_info "🌐 Access your services:"
echo "Dashboard:    minikube service dashboard -n bitcoin-prediction --url"
echo "Web App:      minikube service web-app -n bitcoin-prediction --url"
echo "Kafka UI:     minikube service kafka-ui -n bitcoin-prediction --url"

echo ""
print_info "📋 Quick commands:"
echo "Status:       ./k8s/status.sh"
echo "Logs:         ./k8s/logs.sh <service-name>"
echo "Monitor:      ./k8s/monitor.sh"

echo ""
print_status "🚀 System is ready for use!" 