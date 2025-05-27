#!/bin/bash

# Bitcoin Prediction System - Safe Shutdown
# This script safely stops all services while preserving data

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

# Parse arguments
PAUSE_MINIKUBE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --pause-minikube)
            PAUSE_MINIKUBE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--pause-minikube]"
            echo ""
            echo "Options:"
            echo "  --pause-minikube    Pause minikube to save maximum resources"
            echo ""
            echo "This script safely shuts down all services while preserving:"
            echo "  ✅ Bitcoin price data"
            echo "  ✅ Prediction models"
            echo "  ✅ Kafka data and topics"
            echo "  ✅ Zookeeper data"
            echo "  ✅ Application configurations"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

print_header "🛑 Safely Shutting Down Bitcoin Prediction System"

# Check if minikube is running
if ! minikube status &> /dev/null; then
    print_warning "Minikube is not running"
    exit 0
fi

# Check if namespace exists
if ! kubectl get namespace bitcoin-prediction &> /dev/null; then
    print_warning "Bitcoin prediction namespace not found"
    exit 0
fi

# Show current status
print_info "Current system status:"
kubectl get pods -n bitcoin-prediction 2>/dev/null || echo "No pods found"

print_header "🔄 Scaling Down Services"

# Scale down all deployments to 0 replicas (preserves everything except running pods)
deployments=$(kubectl get deployments -n bitcoin-prediction -o name 2>/dev/null || echo "")

if [ -n "$deployments" ]; then
    print_info "Scaling down deployments to preserve data..."
    
    for deployment in $deployments; do
        deployment_name=$(echo $deployment | cut -d'/' -f2)
        print_info "Stopping $deployment_name..."
        kubectl scale $deployment --replicas=0 -n bitcoin-prediction
    done
    
    print_status "All services stopped"
    
    # Wait for all pods to terminate
    print_info "Waiting for pods to terminate gracefully..."
    kubectl wait --for=delete pod --all -n bitcoin-prediction --timeout=60s 2>/dev/null || print_warning "Some pods may still be terminating"
    
else
    print_warning "No deployments found to scale down"
fi

# Show what's preserved
print_header "💾 Data Preservation Status"

print_info "Checking preserved resources..."

# Check PersistentVolumes
pvcs=$(kubectl get pvc -n bitcoin-prediction --no-headers 2>/dev/null | wc -l || echo "0")
echo "  ✅ PersistentVolumeClaims: $pvcs (data safe)"

# Check ConfigMaps
configmaps=$(kubectl get configmap -n bitcoin-prediction --no-headers 2>/dev/null | wc -l || echo "0")
echo "  ✅ ConfigMaps: $configmaps (configuration safe)"

# Check Services
services=$(kubectl get service -n bitcoin-prediction --no-headers 2>/dev/null | wc -l || echo "0")
echo "  ✅ Services: $services (network config safe)"

# Check Secrets
secrets=$(kubectl get secret -n bitcoin-prediction --no-headers 2>/dev/null | wc -l || echo "0")
echo "  ✅ Secrets: $secrets (credentials safe)"

print_status "All data and configurations preserved"

# Optionally pause minikube
if [ "$PAUSE_MINIKUBE" = true ]; then
    print_header "⏸️ Pausing Minikube"
    print_info "Pausing minikube to save maximum resources..."
    minikube pause
    print_status "Minikube paused"
    
    print_info "To resume later, run: ./k8s/startup.sh"
else
    print_info "Minikube is still running (use --pause-minikube to pause)"
    print_info "To restart services, run: ./k8s/startup.sh"
fi

print_header "✅ Shutdown Complete"

echo "Your Bitcoin Prediction System has been safely shut down."
echo ""
echo "📊 Status:"
echo "  🔴 Services: Stopped"
echo "  ✅ Data: Preserved"
echo "  ✅ Configurations: Preserved"
if [ "$PAUSE_MINIKUBE" = true ]; then
    echo "  ⏸️ Minikube: Paused"
else
    echo "  ✅ Minikube: Running"
fi
echo ""
echo "🚀 To restart:"
echo "  ./k8s/startup.sh"
echo ""
echo "🏗️ To rebuild from scratch:"
echo "  ./k8s/build-from-scratch.sh"
echo ""

print_status "Safe shutdown completed!" 