#!/bin/bash

# Bitcoin Prediction System - Shutdown Script
# Usage: ./shutdown.sh [--pause-minikube|--stop-minikube|--delete-all]

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

# Parse command line arguments
SHUTDOWN_MODE="preserve"
while [[ $# -gt 0 ]]; do
    case $1 in
        --pause-minikube)
            SHUTDOWN_MODE="pause"
            shift
            ;;
        --stop-minikube)
            SHUTDOWN_MODE="stop"
            shift
            ;;
        --delete-all)
            SHUTDOWN_MODE="delete"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Shutdown modes:"
            echo "  (no args)         Preserve data, stop services only"
            echo "  --pause-minikube  Pause minikube (maximum resource saving)"
            echo "  --stop-minikube   Stop minikube completely"
            echo "  --delete-all      Delete everything including data"
            echo ""
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_header "🛑 Shutting Down Bitcoin Prediction System"

# Check if minikube is running
if ! minikube status &> /dev/null; then
    print_warning "Minikube is not running"
    exit 0
fi

case $SHUTDOWN_MODE in
    "preserve")
        print_info "Stopping services while preserving all data..."
        
        # Scale down deployments to 0 replicas
        print_info "Scaling down application services..."
        kubectl scale deployment/data-collector --replicas=0 -n bitcoin-prediction 2>/dev/null || true
        kubectl scale deployment/bitcoin-forecast-app --replicas=0 -n bitcoin-prediction 2>/dev/null || true
        kubectl scale deployment/web-app --replicas=0 -n bitcoin-prediction 2>/dev/null || true
        kubectl scale deployment/kafka-ui --replicas=0 -n bitcoin-prediction 2>/dev/null || true
        
        print_info "Scaling down infrastructure services..."
        kubectl scale deployment/kafka --replicas=0 -n bitcoin-prediction 2>/dev/null || true
        kubectl scale deployment/zookeeper --replicas=0 -n bitcoin-prediction 2>/dev/null || true
        
        print_status "All services stopped, data preserved"
        print_info "To restart: ./k8s/startup.sh"
        ;;
        
    "pause")
        print_info "Pausing minikube (maximum resource saving)..."
        
        # First scale down services
        kubectl scale deployment --all --replicas=0 -n bitcoin-prediction 2>/dev/null || true
        
        # Pause minikube
        minikube pause
        
        print_status "Minikube paused, all data preserved"
        print_info "To resume: ./k8s/startup.sh"
        ;;
        
    "stop")
        print_info "Stopping minikube completely..."
        minikube stop
        
        print_status "Minikube stopped, data preserved in volumes"
        print_info "To restart: ./k8s/build-from-scratch.sh (faster restart)"
        ;;
        
    "delete")
        print_warning "⚠️  DELETING ALL DATA - This cannot be undone!"
        read -p "Are you sure? Type 'yes' to confirm: " confirm
        if [ "$confirm" = "yes" ]; then
            print_info "Deleting namespace and all data..."
            kubectl delete namespace bitcoin-prediction 2>/dev/null || true
            
            print_info "Stopping minikube..."
            minikube stop
            
            print_info "Deleting minikube cluster..."
            minikube delete
            
            print_status "Everything deleted - fresh start required"
            print_info "To start fresh: ./k8s/build-from-scratch.sh"
        else
            print_info "Deletion cancelled"
        fi
        ;;
esac

print_header "✅ Shutdown Complete" 