#!/bin/bash

# Bitcoin Prediction System - Create Persistent Tunnels
# This script creates background tunnels with fixed localhost URLs

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

# Check if minikube is running
if ! minikube status &> /dev/null; then
    print_error "Minikube is not running. Run: ./k8s/startup.sh"
    exit 1
fi

# Check if namespace exists
if ! kubectl get namespace bitcoin-prediction &> /dev/null; then
    print_error "Bitcoin prediction namespace not found. Run: ./k8s/build-from-scratch.sh"
    exit 1
fi

print_header "🚇 Creating Persistent Service Tunnels"

# PID file to track background processes
PID_FILE="/tmp/bitcoin-prediction-tunnels.pid"

# Function to kill existing tunnels
kill_existing_tunnels() {
    if [ -f "$PID_FILE" ]; then
        print_info "Stopping existing tunnels..."
        while read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done < "$PID_FILE"
        rm -f "$PID_FILE"
        print_status "Existing tunnels stopped"
    fi
}

# Function to check if port is available
is_port_available() {
    local port=$1
    ! lsof -i :$port &> /dev/null
}

# Function to create tunnel
create_tunnel() {
    local service=$1
    local local_port=$2
    local service_port=$3
    
    if ! is_port_available $local_port; then
        print_warning "Port $local_port is already in use, skipping $service"
        return 1
    fi
    
    print_info "Creating tunnel for $service (localhost:$local_port)"
    
    # Create port-forward in background with output redirected to /dev/null
    kubectl port-forward service/$service $local_port:$service_port -n bitcoin-prediction > /dev/null 2>&1 &
    local pid=$!
    
    # Save PID for cleanup
    echo $pid >> "$PID_FILE"
    
    # Wait a moment for tunnel to establish
    sleep 2
    
    # Test if tunnel is working
    if kill -0 $pid 2>/dev/null; then
        print_status "$service tunnel ready at http://localhost:$local_port"
        return 0
    else
        print_error "Failed to create tunnel for $service"
        return 1
    fi
}

# Parse command line arguments
ACTION="start"
while [[ $# -gt 0 ]]; do
    case $1 in
        --stop)
            ACTION="stop"
            shift
            ;;
        --restart)
            ACTION="restart"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--stop|--restart]"
            echo ""
            echo "Creates persistent background tunnels for Bitcoin Prediction services:"
            echo "  Web App:    http://localhost:5001"
            echo "  Dashboard:  http://localhost:8501"
            echo "  Kafka UI:   http://localhost:8080"
            echo ""
            echo "Options:"
            echo "  --stop      Stop all tunnels"
            echo "  --restart   Restart all tunnels"
            echo "  (no args)   Start tunnels"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Handle stop action
if [ "$ACTION" = "stop" ]; then
    print_header "🛑 Stopping All Tunnels"
    kill_existing_tunnels
    print_status "All tunnels stopped"
    exit 0
fi

# Handle restart action
if [ "$ACTION" = "restart" ]; then
    kill_existing_tunnels
    sleep 1
fi

# Kill existing tunnels before starting new ones
kill_existing_tunnels

# Create tunnels for each service
print_info "Creating background tunnels with fixed localhost URLs..."
echo ""

# Web App (port 5001 instead of 5000 to avoid macOS ControlCenter)
create_tunnel "web-app" "5001" "5000"

# Dashboard (port 8501)  
create_tunnel "dashboard" "8501" "8501"

# Kafka UI (port 8080)
create_tunnel "kafka-ui" "8080" "8080"

echo ""

# Show the fixed URLs
print_header "🌐 Fixed Service URLs (Always the Same!)"
echo "🌐 Web App:     http://localhost:5001"
echo "📊 Dashboard:   http://localhost:8501"
echo "⚙️  Kafka UI:    http://localhost:8080"
echo ""

# Test accessibility
print_header "🔍 Testing Fixed URLs"

test_localhost_url() {
    local name=$1
    local port=$2
    local url="http://localhost:$port"
    echo -n "Testing $name ($url): "
    
    # Give service a moment to respond
    sleep 1
    
    if curl -s --connect-timeout 3 --max-time 5 "$url" > /dev/null 2>&1; then
        print_status "✅ Working"
        return 0
    else
        print_warning "❌ Not responding (may be starting up)"
        return 1
    fi
}

test_localhost_url "Web App" "5001"
test_localhost_url "Dashboard" "8501"
test_localhost_url "Kafka UI" "8080"

echo ""

# Show management commands
print_header "🔧 Tunnel Management"
echo "📋 Commands:"
echo "  Stop tunnels:    ./k8s/create-tunnels.sh --stop"
echo "  Restart tunnels: ./k8s/create-tunnels.sh --restart"
echo "  Show tunnels:    ps aux | grep 'kubectl port-forward'"
echo ""

print_status "✅ Persistent tunnels created successfully!"
print_info "💡 These URLs are now fixed and work until you stop the tunnels"
print_info "💾 Tunnel PIDs saved to: $PID_FILE"

# Show how to stop tunnels on exit
echo ""
print_warning "💡 Important: To stop tunnels when done, run:"
echo "    ./k8s/create-tunnels.sh --stop" 