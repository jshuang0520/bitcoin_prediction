#!/bin/bash

# Bitcoin Prediction System - Startup Script
# Resume system after shutdown while preserving all data

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

# Check if Docker is running
if ! docker info &> /dev/null; then
    print_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if minikube exists and handle different states
print_info "Checking minikube status..."

if ! command -v minikube &> /dev/null; then
    print_error "minikube is not installed. Install with: brew install minikube"
    exit 1
fi

# Handle different minikube states
if minikube status &> /dev/null; then
    # Check if it's paused
    if minikube status | grep -q "Paused"; then
        print_info "Resuming paused minikube..."
        minikube unpause
        print_status "Minikube resumed"
    elif docker ps --format "table {{.Names}}" | grep -q "^minikube$"; then
        print_status "Minikube is already running"
    else
        print_warning "Minikube state corrupted, fixing..."
        minikube delete
        minikube start --driver=docker --memory=6144 --cpus=4
        print_status "Minikube restarted"
    fi
else
    print_info "Starting minikube..."
    minikube start --driver=docker --memory=6144 --cpus=4
    print_status "Minikube started"
fi

# Update context to ensure we're pointing to the right cluster
minikube update-context
print_status "Kubectl context updated"

# Check if namespace exists
if ! kubectl get namespace bitcoin-prediction &> /dev/null; then
    print_error "Bitcoin prediction namespace not found."
    print_info "It looks like the system was not deployed yet or was completely deleted."
    print_info "Run: ./k8s/build-from-scratch.sh"
    exit 1
fi

print_header "🔄 Restoring Services"

# Restore original replica counts for infrastructure
print_info "Starting infrastructure services..."
kubectl scale deployment/zookeeper --replicas=1 -n bitcoin-prediction
kubectl scale deployment/kafka --replicas=1 -n bitcoin-prediction

# Wait for infrastructure to be ready
print_info "Waiting for infrastructure to be ready..."
kubectl wait --for=condition=available --timeout=120s deployment/zookeeper -n bitcoin-prediction
kubectl wait --for=condition=available --timeout=180s deployment/kafka -n bitcoin-prediction
print_status "Infrastructure services ready"

# Restore application services
print_info "Starting application services..."
kubectl scale deployment/data-collector --replicas=1 -n bitcoin-prediction &
kubectl scale deployment/bitcoin-forecast-app --replicas=1 -n bitcoin-prediction &
kubectl scale deployment/web-app --replicas=1 -n bitcoin-prediction &
kubectl scale deployment/kafka-ui --replicas=1 -n bitcoin-prediction &
wait

# Wait for all services to be ready
print_info "Waiting for all services to be ready..."
{
    kubectl wait --for=condition=available --timeout=300s deployment/data-collector -n bitcoin-prediction &
    kubectl wait --for=condition=available --timeout=300s deployment/bitcoin-forecast-app -n bitcoin-prediction &
    kubectl wait --for=condition=available --timeout=300s deployment/web-app -n bitcoin-prediction &
    kubectl wait --for=condition=available --timeout=300s deployment/kafka-ui -n bitcoin-prediction &
    wait
} && print_status "All services ready!" || print_warning "Some services may still be starting up"

# Show deployment status
print_header "📊 System Status"
kubectl get pods -n bitcoin-prediction

# Get minikube IP for access URLs
MINIKUBE_IP=$(minikube ip)

# Show access information
print_header "🌐 Service Access"
echo "Your Bitcoin Prediction System is now running!"
echo ""
echo "Service URLs:"
echo "  Web App:    http://$MINIKUBE_IP:30001"
echo "  Kafka UI:   http://$MINIKUBE_IP:30003"
echo ""

# Test web-app accessibility
print_info "Testing service accessibility..."
if curl -s --connect-timeout 5 http://$MINIKUBE_IP:30001 > /dev/null; then
    print_status "Web-app is accessible"
else
    print_warning "Web-app may still be starting up"
fi

print_header "✅ Startup Complete"
echo "Your Bitcoin prediction system has been restored!"
echo ""
echo "📋 Next steps:"
echo "1. Create fixed localhost URLs: ./k8s/create-tunnels.sh"
echo "2. Monitor system: ./k8s/status.sh"
echo "3. View logs: ./k8s/logs.sh <service-name>"
echo ""

print_status "System startup completed successfully!" 