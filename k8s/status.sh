#!/bin/bash

# Bitcoin Prediction System - Status Check Script
# This script provides comprehensive status information about the Kubernetes deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_status() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

# Check if minikube is running
if ! minikube status &> /dev/null; then
    print_error "minikube is not running"
    echo "Start with: ./k8s/start-minikube.sh"
    exit 1
fi

print_header "BITCOIN PREDICTION SYSTEM STATUS"

# Show cluster info
print_header "Cluster Information"
kubectl cluster-info
echo ""

# Show namespace status
print_header "Namespace Status"
kubectl get namespace bitcoin-prediction 2>/dev/null || print_warning "Namespace bitcoin-prediction not found"
echo ""

# Show all pods with detailed status
print_header "Pod Status"
kubectl get pods -n bitcoin-prediction -o wide
echo ""

# Show services
print_header "Service Status"
kubectl get services -n bitcoin-prediction
echo ""

# Show persistent volume claims
print_header "Storage Status"
kubectl get pvc -n bitcoin-prediction
echo ""

# Show resource usage if metrics-server is available
print_header "Resource Usage"
kubectl top pods -n bitcoin-prediction 2>/dev/null || print_warning "Metrics not available (metrics-server may not be running)"
echo ""

# Show recent events
print_header "Recent Events"
kubectl get events -n bitcoin-prediction --sort-by='.lastTimestamp' | tail -10
echo ""

# Check individual service health
print_header "Service Health Check"

services=("zookeeper" "kafka" "data-collector" "bitcoin-forecast-app" "web-app" "kafka-ui")

for service in "${services[@]}"; do
    echo -n "Checking $service: "
    if kubectl get deployment $service -n bitcoin-prediction &> /dev/null; then
        ready=$(kubectl get deployment $service -n bitcoin-prediction -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "0")
        desired=$(kubectl get deployment $service -n bitcoin-prediction -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "1")
        if [ "$ready" = "$desired" ] && [ "$ready" != "0" ]; then
            print_status "Ready ($ready/$desired)"
        else
            print_warning "Not Ready ($ready/$desired)"
        fi
    else
        print_error "Not Found"
    fi
done

echo ""

# Show access URLs
print_header "Access URLs"
echo "To access your services, run these commands:"
echo ""
echo "Web App:      minikube service web-app -n bitcoin-prediction"
echo "Kafka UI:     minikube service kafka-ui -n bitcoin-prediction"
echo ""

# Show useful commands
print_header "Useful Commands"
echo "View logs:           kubectl logs -f deployment/<service-name> -n bitcoin-prediction"
echo "Restart service:     kubectl rollout restart deployment/<service-name> -n bitcoin-prediction"
echo "Scale service:       kubectl scale deployment/<service-name> --replicas=<number> -n bitcoin-prediction"
echo "Shell into pod:      kubectl exec -it deployment/<service-name> -n bitcoin-prediction -- bash"
echo "Port forward:        kubectl port-forward service/<service-name> <local-port>:<service-port> -n bitcoin-prediction"
echo ""

print_status "Status check completed!" 