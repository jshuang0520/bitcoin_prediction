#!/bin/bash

# Bitcoin Prediction System - Service Update Script
# Usage: ./update-service.sh <service-name>
# Example: ./update-service.sh bitcoin-forecast-app

set -e

SERVICE_NAME=$1

if [ -z "$SERVICE_NAME" ]; then
    echo "❌ Usage: ./update-service.sh <service-name>"
    echo ""
    echo "Available services:"
    echo "  - bitcoin-forecast-app"
    echo "  - data-collector"
    echo "  - web-app"
    echo "  - dashboard"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

echo "🔄 Updating service: $SERVICE_NAME"

# Check if minikube is running
if ! minikube status &> /dev/null; then
    print_error "minikube is not running. Start it with: minikube start --driver=docker"
    exit 1
fi

# Configure Docker environment for minikube
print_warning "Configuring Docker environment for minikube..."
eval $(minikube docker-env)

# Navigate to project root
cd "$(dirname "$0")/.."

# Build the specific service image
case $SERVICE_NAME in
    "bitcoin-forecast-app")
        echo "🔨 Building bitcoin-forecast-app..."
        docker build -f bitcoin_forecast_app/Dockerfile -t bitcoin-forecast-app:latest .
        ;;
    "data-collector")
        echo "🔨 Building data-collector..."
        docker build -f data_collector/Dockerfile -t data-collector:latest .
        ;;
    "web-app")
        echo "🔨 Building web-app..."
        docker build -f web_app/Dockerfile -t web-app:latest ./web_app
        ;;
    "dashboard")
        echo "🔨 Building dashboard..."
        docker build -f dashboard/Dockerfile -t dashboard:latest .
        ;;
    *)
        print_error "Unknown service: $SERVICE_NAME"
        echo "Available services: bitcoin-forecast-app, data-collector, web-app, dashboard"
        exit 1
        ;;
esac

print_status "Image built successfully"

# Restart the deployment in Kubernetes
echo "🚀 Restarting deployment in Kubernetes..."
kubectl rollout restart deployment/$SERVICE_NAME -n bitcoin-prediction

# Wait for rollout to complete
echo "⏳ Waiting for rollout to complete..."
kubectl rollout status deployment/$SERVICE_NAME -n bitcoin-prediction --timeout=300s

print_status "Service $SERVICE_NAME updated successfully!"

# Show pod status
echo ""
echo "📊 Current pod status:"
kubectl get pods -n bitcoin-prediction -l app=$SERVICE_NAME

echo ""
echo "🔍 To view logs:"
echo "  kubectl logs -f deployment/$SERVICE_NAME -n bitcoin-prediction"

# If it's a user-facing service, show access info
case $SERVICE_NAME in
    "web-app")
        echo ""
        echo "🌐 Access web-app:"
        echo "  minikube service web-app -n bitcoin-prediction"
        ;;
    "dashboard")
        echo ""
        echo "🌐 Access dashboard:"
        echo "  minikube service dashboard -n bitcoin-prediction"
        ;;
esac 