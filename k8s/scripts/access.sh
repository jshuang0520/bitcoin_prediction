#!/bin/bash

# Bitcoin Prediction System - Service Access Script
# This script provides non-blocking access to services

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

# Function to show usage
show_usage() {
    echo "Usage: $0 [service-name] [options]"
    echo ""
    echo "Available services:"
    echo "  dashboard     - Streamlit dashboard with Bitcoin predictions"
    echo "  web-app       - FastAPI web application"
    echo "  kafka-ui      - Kafka management interface"
    echo ""
    echo "Options:"
    echo "  --url         - Get URL only (non-blocking)"
    echo "  --open        - Open in browser (background)"
    echo "  --tunnel      - Create port tunnel (background)"
    echo ""
    echo "Examples:"
    echo "  $0 dashboard --url                    # Get dashboard URL"
    echo "  $0 dashboard --open                   # Open dashboard in browser"
    echo "  $0 web-app --tunnel                   # Create port tunnel for web-app"
    echo "  $0 --all --url                       # Get all service URLs"
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
ACTION=""
SHOW_ALL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --url)
            ACTION="url"
            shift
            ;;
        --open)
            ACTION="open"
            shift
            ;;
        --tunnel)
            ACTION="tunnel"
            shift
            ;;
        --all)
            SHOW_ALL="true"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        dashboard|web-app|kafka-ui)
            SERVICE="$1"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Function to get service URL
get_service_url() {
    local service=$1
    minikube service $service -n bitcoin-prediction --url 2>/dev/null
}

# Function to open service in browser
open_service() {
    local service=$1
    print_info "Opening $service in browser..."
    
    # Get URL and open in background
    local url=$(get_service_url $service)
    if [[ -n "$url" ]]; then
        print_status "Opening $url"
        if command -v open &> /dev/null; then
            open "$url" &
        elif command -v xdg-open &> /dev/null; then
            xdg-open "$url" &
        else
            print_warning "Cannot open browser automatically. URL: $url"
        fi
    else
        print_error "Could not get URL for $service"
    fi
}

# Function to create port tunnel
create_tunnel() {
    local service=$1
    local port=""
    
    case $service in
        dashboard)
            port="8501"
            ;;
        web-app)
            port="5000"
            ;;
        kafka-ui)
            port="8080"
            ;;
    esac
    
    if [[ -n "$port" ]]; then
        print_info "Creating port tunnel for $service on localhost:$port"
        print_warning "Press Ctrl+C to stop the tunnel"
        kubectl port-forward service/$service $port:$port -n bitcoin-prediction
    else
        print_error "Unknown port for service $service"
    fi
}

# Function to show all URLs
show_all_urls() {
    echo ""
    print_info "🌐 Service URLs:"
    echo ""
    
    services=("dashboard" "web-app" "kafka-ui")
    descriptions=("Bitcoin Predictions Dashboard" "FastAPI Web Application" "Kafka Management UI")
    
    for i in "${!services[@]}"; do
        service="${services[$i]}"
        desc="${descriptions[$i]}"
        url=$(get_service_url $service 2>/dev/null || echo "Not available")
        printf "%-30s %s\n" "$desc:" "$url"
    done
    
    echo ""
    print_info "💡 Tips:"
    echo "  • Copy URLs to access in browser"
    echo "  • Use --open to open automatically"
    echo "  • Use --tunnel for localhost access"
}

# Main logic
if [[ "$SHOW_ALL" == "true" ]]; then
    show_all_urls
elif [[ -z "$SERVICE" ]]; then
    print_error "Please specify a service name or use --all"
    echo ""
    show_usage
    exit 1
else
    case $ACTION in
        url)
            url=$(get_service_url $SERVICE)
            if [[ -n "$url" ]]; then
                echo "$url"
            else
                print_error "Could not get URL for $SERVICE"
                exit 1
            fi
            ;;
        open)
            open_service $SERVICE
            ;;
        tunnel)
            create_tunnel $SERVICE
            ;;
        *)
            # Default: show URL
            url=$(get_service_url $SERVICE)
            if [[ -n "$url" ]]; then
                print_info "$SERVICE is available at: $url"
            else
                print_error "Could not get URL for $SERVICE"
                exit 1
            fi
            ;;
    esac
fi 