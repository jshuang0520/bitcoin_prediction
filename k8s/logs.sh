#!/bin/bash

# Bitcoin Prediction System - Logs Viewer Script
# This script helps you view logs from any service

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

# Available services
services=("zookeeper" "kafka" "data-collector" "bitcoin-forecast-app" "web-app" "kafka-ui")

# Function to show usage
show_usage() {
    echo "Usage: $0 [service-name] [options]"
    echo ""
    echo "Available services:"
    for service in "${services[@]}"; do
        echo "  - $service"
    done
    echo ""
    echo "Options:"
    echo "  -f, --follow     Follow log output (like tail -f)"
    echo "  -t, --tail N     Show last N lines (default: 100)"
    echo "  --all           Show logs from all services"
    echo ""
    echo "Examples:"
    echo "  $0 bitcoin-forecast-app              # Show last 100 lines"
    echo "  $0 bitcoin-forecast-app -f           # Follow logs"
    echo "  $0 data-collector --tail 50          # Show last 50 lines"
    echo "  $0 --all                            # Show logs from all services"
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

# Parse arguments
SERVICE=""
FOLLOW=""
TAIL="100"
SHOW_ALL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--follow)
            FOLLOW="-f"
            shift
            ;;
        -t|--tail)
            TAIL="$2"
            shift 2
            ;;
        --all)
            SHOW_ALL="true"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            if [[ -z "$SERVICE" ]]; then
                SERVICE="$1"
            else
                print_error "Unknown option: $1"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Function to show logs for a service
show_service_logs() {
    local service=$1
    local follow=$2
    local tail=$3
    
    print_info "Showing logs for $service (last $tail lines)..."
    echo "Press Ctrl+C to stop"
    echo "----------------------------------------"
    
    if kubectl get deployment $service -n bitcoin-prediction &> /dev/null; then
        kubectl logs deployment/$service -n bitcoin-prediction --tail=$tail $follow
    else
        print_error "Service $service not found"
        return 1
    fi
}

# Function to show all logs
show_all_logs() {
    for service in "${services[@]}"; do
        echo ""
        print_info "=== $service ==="
        if kubectl get deployment $service -n bitcoin-prediction &> /dev/null; then
            kubectl logs deployment/$service -n bitcoin-prediction --tail=20
        else
            print_warning "Service $service not found"
        fi
        echo ""
    done
}

# Main logic
if [[ "$SHOW_ALL" == "true" ]]; then
    show_all_logs
elif [[ -z "$SERVICE" ]]; then
    print_error "Please specify a service name or use --all"
    echo ""
    show_usage
    exit 1
else
    # Validate service name
    if [[ ! " ${services[@]} " =~ " ${SERVICE} " ]]; then
        print_error "Invalid service name: $SERVICE"
        echo ""
        show_usage
        exit 1
    fi
    
    show_service_logs "$SERVICE" "$FOLLOW" "$TAIL"
fi 