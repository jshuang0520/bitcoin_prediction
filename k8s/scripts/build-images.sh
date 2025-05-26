#!/bin/bash

# Bitcoin Prediction System - Docker Image Build Script
# This script builds all required Docker images for Kubernetes deployment

set -e

echo "🔨 Building Docker images for Bitcoin Prediction System..."

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

# Check if we're using minikube
if command -v minikube &> /dev/null && minikube status &> /dev/null; then
    print_warning "Detected minikube - configuring Docker environment"
    eval $(minikube docker-env)
    print_status "Using minikube Docker daemon"
fi

# Navigate to project root
cd "$(dirname "$0")/.."

echo "📊 Building application images..."

# Build data-collector
echo "Building data-collector..."
docker build -f data_collector/Dockerfile -t data-collector:latest .
print_status "data-collector image built"

# Build bitcoin-forecast-app
echo "Building bitcoin-forecast-app..."
docker build -f bitcoin_forecast_app/Dockerfile -t bitcoin-forecast-app:latest .
print_status "bitcoin-forecast-app image built"

# Build web-app
echo "Building web-app..."
docker build -f web_app/Dockerfile -t web-app:latest .
print_status "web-app image built"

# Build dashboard
echo "Building dashboard..."
docker build -f dashboard/Dockerfile -t dashboard:latest .
print_status "dashboard image built"

echo ""
print_status "All images built successfully!"

# Display built images
echo ""
echo "📦 Built Images:"
docker images | grep -E "(data-collector|bitcoin-forecast-app|web-app|dashboard)" | grep latest

echo ""
echo "🚀 Ready for Kubernetes deployment!"
echo "Run: cd k8s && ./deploy.sh" 