#!/bin/bash

# Bitcoin Prediction System - Cleanup Script
# This script properly stops and cleans up the Kubernetes deployment

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

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --apps-only     Delete only application deployments (keep infrastructure)"
    echo "  --all          Delete everything including namespace and storage"
    echo "  --stop-minikube Stop minikube completely"
    echo "  --help         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Delete all deployments but keep storage"
    echo "  $0 --apps-only       # Delete only app deployments (keep Kafka/Zookeeper)"
    echo "  $0 --all             # Delete everything including storage"
    echo "  $0 --stop-minikube   # Stop minikube completely"
}

# Parse arguments
APPS_ONLY=""
DELETE_ALL=""
STOP_MINIKUBE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --apps-only)
            APPS_ONLY="true"
            shift
            ;;
        --all)
            DELETE_ALL="true"
            shift
            ;;
        --stop-minikube)
            STOP_MINIKUBE="true"
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

print_info "Starting cleanup process..."

# Function to delete application deployments only
cleanup_apps_only() {
    print_info "Deleting application deployments only..."
    
    apps=("data-collector" "bitcoin-forecast-app" "web-app" "dashboard" "kafka-ui")
    
    for app in "${apps[@]}"; do
        if kubectl get deployment $app -n bitcoin-prediction &> /dev/null; then
            kubectl delete deployment $app -n bitcoin-prediction
            print_status "Deleted $app deployment"
        else
            print_warning "$app deployment not found"
        fi
    done
    
    # Delete app services
    app_services=("web-app" "dashboard" "kafka-ui")
    for service in "${app_services[@]}"; do
        if kubectl get service $service -n bitcoin-prediction &> /dev/null; then
            kubectl delete service $service -n bitcoin-prediction
            print_status "Deleted $service service"
        fi
    done
    
    print_status "Application cleanup completed"
}

# Function to delete all deployments
cleanup_deployments() {
    print_info "Deleting all deployments..."
    
    if kubectl get namespace bitcoin-prediction &> /dev/null; then
        # Delete all deployments
        kubectl delete deployments --all -n bitcoin-prediction
        print_status "Deleted all deployments"
        
        # Delete all services
        kubectl delete services --all -n bitcoin-prediction
        print_status "Deleted all services"
        
        # Delete jobs
        kubectl delete jobs --all -n bitcoin-prediction 2>/dev/null || true
        print_status "Deleted all jobs"
        
        # Delete HPA
        kubectl delete hpa --all -n bitcoin-prediction 2>/dev/null || true
        print_status "Deleted all HPAs"
        
        # Delete resource quotas
        kubectl delete resourcequota --all -n bitcoin-prediction 2>/dev/null || true
        print_status "Deleted all resource quotas"
    else
        print_warning "Namespace bitcoin-prediction not found"
    fi
}

# Function to delete everything including storage
cleanup_all() {
    print_info "Deleting everything including storage..."
    
    if kubectl get namespace bitcoin-prediction &> /dev/null; then
        kubectl delete namespace bitcoin-prediction
        print_status "Deleted namespace bitcoin-prediction (this will delete everything)"
        
        # Wait for namespace to be fully deleted
        print_info "Waiting for namespace to be fully deleted..."
        while kubectl get namespace bitcoin-prediction &> /dev/null; do
            sleep 2
        done
        print_status "Namespace fully deleted"
    else
        print_warning "Namespace bitcoin-prediction not found"
    fi
}

# Function to stop minikube
stop_minikube() {
    print_info "Stopping minikube..."
    
    if command -v minikube &> /dev/null; then
        if minikube status &> /dev/null; then
            minikube stop
            print_status "Minikube stopped"
        else
            print_warning "Minikube is not running"
        fi
    else
        print_error "Minikube is not installed"
    fi
}

# Main cleanup logic
if [[ "$STOP_MINIKUBE" == "true" ]]; then
    stop_minikube
elif [[ "$DELETE_ALL" == "true" ]]; then
    cleanup_all
elif [[ "$APPS_ONLY" == "true" ]]; then
    cleanup_apps_only
else
    # Default: delete deployments but keep storage
    cleanup_deployments
fi

print_status "Cleanup completed!"

# Show what's left
if [[ "$STOP_MINIKUBE" != "true" ]] && [[ "$DELETE_ALL" != "true" ]]; then
    echo ""
    print_info "Remaining resources:"
    kubectl get all -n bitcoin-prediction 2>/dev/null || print_warning "No resources found"
    
    if [[ "$APPS_ONLY" != "true" ]]; then
        echo ""
        print_info "Storage (PVCs) still exists:"
        kubectl get pvc -n bitcoin-prediction 2>/dev/null || print_warning "No PVCs found"
    fi
fi

echo ""
print_info "To restart the system:"
echo "  ./k8s/deploy.sh                    # Redeploy everything"
 