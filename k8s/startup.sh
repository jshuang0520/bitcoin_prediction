#!/bin/bash

# Bitcoin Prediction System - Quick Startup
# This script restarts the system after shutdown without rebuilding

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

print_header "🚀 Starting Bitcoin Prediction System"

# Check if minikube is installed
if ! command -v minikube &> /dev/null; then
    print_error "minikube is not installed. Please install minikube and try again."
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed. Please install kubectl and try again."
    exit 1
fi

# Check minikube status and resume if paused
print_info "Checking minikube status..."
if minikube status | grep -q "Paused"; then
    print_info "Minikube is paused, resuming..."
    minikube unpause
    print_status "Minikube resumed"
elif ! minikube status &> /dev/null; then
    print_info "Starting minikube..."
    minikube start --driver=docker
    print_status "Minikube started"
else
    print_status "Minikube is already running"
fi

# Update context
minikube update-context
print_status "Kubectl context updated"

# Check if namespace exists
if ! kubectl get namespace bitcoin-prediction &> /dev/null; then
    print_error "Bitcoin prediction namespace not found. Please run ./k8s/build-from-scratch.sh first."
    exit 1
fi

print_header "🔄 Scaling Up Services"

# Get current deployment status
print_info "Checking current deployment status..."
deployments=$(kubectl get deployments -n bitcoin-prediction -o name 2>/dev/null || echo "")

if [ -z "$deployments" ]; then
    print_error "No deployments found. Please run ./k8s/build-from-scratch.sh first."
    exit 1
fi

# Scale up deployments in dependency order
print_info "Starting services in dependency order..."

# Start Zookeeper first
print_info "Starting Zookeeper..."
kubectl scale deployment/zookeeper --replicas=1 -n bitcoin-prediction
kubectl wait --for=condition=available --timeout=120s deployment/zookeeper -n bitcoin-prediction
print_status "Zookeeper ready"

# Start Kafka
print_info "Starting Kafka..."
kubectl scale deployment/kafka --replicas=1 -n bitcoin-prediction
kubectl wait --for=condition=available --timeout=180s deployment/kafka -n bitcoin-prediction
print_status "Kafka ready"

# Start application services in parallel
print_info "Starting application services..."
kubectl scale deployment/data-collector --replicas=1 -n bitcoin-prediction &
kubectl scale deployment/bitcoin-forecast-app --replicas=1 -n bitcoin-prediction &
kubectl scale deployment/web-app --replicas=1 -n bitcoin-prediction &
kubectl scale deployment/dashboard --replicas=1 -n bitcoin-prediction &
kubectl scale deployment/kafka-ui --replicas=1 -n bitcoin-prediction &
wait

print_status "All services started"

# Wait for all services to be ready
print_info "Waiting for all services to be ready..."
{
    kubectl wait --for=condition=available --timeout=300s deployment/data-collector -n bitcoin-prediction &
    kubectl wait --for=condition=available --timeout=300s deployment/bitcoin-forecast-app -n bitcoin-prediction &
    kubectl wait --for=condition=available --timeout=300s deployment/web-app -n bitcoin-prediction &
    kubectl wait --for=condition=available --timeout=300s deployment/dashboard -n bitcoin-prediction &
    kubectl wait --for=condition=available --timeout=300s deployment/kafka-ui -n bitcoin-prediction &
    wait
} && print_status "All services are ready!" || print_warning "Some services may still be starting up"

# Show current status
print_header "📊 System Status"
kubectl get pods -n bitcoin-prediction

# Get minikube IP for URLs
MINIKUBE_IP=$(minikube ip)

# Show access information
print_header "🌐 Access Information"
echo "Your Bitcoin Prediction System is now running!"
echo ""
echo "📱 Fixed Service URLs:"
echo "  Web App:    http://$MINIKUBE_IP:30001"
echo "  Dashboard:  http://$MINIKUBE_IP:30002"
echo "  Kafka UI:   http://$MINIKUBE_IP:30003"
echo ""

# Test web-app accessibility
print_info "Testing web-app accessibility..."
sleep 5  # Give services a moment to fully start
if curl -s --connect-timeout 5 http://$MINIKUBE_IP:30001 > /dev/null; then
    print_status "Web-app is accessible"
else
    print_warning "Web-app may still be starting up. Please wait a moment."
fi

print_header "✅ Startup Complete"

echo "Your Bitcoin Prediction System is back online!"
echo ""
echo "📋 Quick Management Commands:"
echo "  Status:     ./k8s/status.sh"
echo "  Logs:       ./k8s/logs.sh <service-name>"
echo "  Update:     ./k8s/update-service.sh <service-name>"
echo "  Shutdown:   ./k8s/shutdown.sh"
echo ""
echo "💡 Tips:"
echo "  - All your data has been preserved"
echo "  - Services use the same fixed URLs as before"
echo "  - Use update-service.sh for quick code changes"
echo ""

print_status "Startup completed successfully!" 