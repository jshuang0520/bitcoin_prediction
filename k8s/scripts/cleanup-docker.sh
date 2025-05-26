#!/bin/bash

# Docker Image Cleanup Script
# Removes redundant and unused Docker images

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

echo "🧹 Docker Image Cleanup"
echo "======================"

# Show current images
echo ""
print_info "Current Docker images:"
docker images

echo ""
print_warning "This will remove:"
echo "  - Dangling images (untagged)"
echo "  - Unused images not referenced by containers"
echo "  - Build cache"

read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
print_info "Removing dangling images..."
docker image prune -f

echo ""
print_info "Removing unused images..."
docker image prune -a -f

echo ""
print_info "Removing build cache..."
docker builder prune -f

echo ""
print_status "Cleanup completed!"

echo ""
print_info "Remaining Docker images:"
docker images

echo ""
print_info "💡 To avoid redundant kicbase images in the future:"
echo "  - Use 'minikube delete' before 'minikube start'"
echo "  - Or use 'minikube start --force' to reuse existing VM" 