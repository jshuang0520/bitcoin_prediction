#!/bin/bash

# Bitcoin Prediction System - Minikube Startup Script
# This script properly starts minikube for the Bitcoin prediction system

set -e

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

echo "🚀 Starting minikube for Bitcoin Prediction System..."

# Check if Docker Desktop is running
if ! docker info &> /dev/null; then
    print_error "Docker Desktop is not running"
    echo "Please start Docker Desktop and try again"
    exit 1
fi

# Check if minikube is installed
if ! command -v minikube &> /dev/null; then
    print_error "minikube is not installed"
    echo "Install with: brew install minikube"
    exit 1
fi

# Stop any existing minikube instance
print_warning "Stopping any existing minikube instance..."
minikube stop &> /dev/null || true

# Delete existing minikube instance for clean start
print_warning "Deleting existing minikube instance for clean start..."
minikube delete &> /dev/null || true

# Start minikube with proper resources
echo "🔄 Starting minikube (this may take a few minutes)..."
minikube start \
    --driver=docker \
    --memory=6144 \
    --cpus=4 \
    --disk-size=20g \
    --kubernetes-version=v1.28.3

# Wait for cluster to be fully ready
echo "⏳ Waiting for cluster to be ready..."
kubectl wait --for=condition=Ready nodes --all --timeout=300s

print_status "minikube started successfully!"

# Enable required addons
echo "🔧 Enabling required addons..."
minikube addons enable metrics-server
minikube addons enable storage-provisioner
minikube addons enable default-storageclass

print_status "Addons enabled"

# Verify cluster is working
echo "🔍 Verifying cluster..."
kubectl cluster-info
kubectl get nodes

print_status "Cluster is ready!"

echo ""
echo "🎯 Next steps:"
echo "1. Configure Docker environment: eval \$(minikube docker-env)"
echo "2. Build images: ./k8s/build-images.sh"
echo "3. Deploy system: ./k8s/deploy.sh"
echo ""
echo "Or run all at once:"
echo "  eval \$(minikube docker-env) && ./k8s/build-images.sh && ./k8s/deploy.sh" 