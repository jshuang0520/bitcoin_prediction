#!/bin/bash

# Bitcoin Prediction System - Monitoring Script
# This script provides background monitoring capabilities similar to docker-compose logs

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
    echo "  bitcoin-forecast-app  - ML prediction service"
    echo "  data-collector       - Bitcoin price collector"
    echo "  web-app             - FastAPI web application"
    echo "  dashboard           - Streamlit dashboard"
    echo "  kafka               - Kafka message broker"
    echo "  kafka-ui            - Kafka management UI"
    echo "  zookeeper           - Zookeeper coordination"
    echo ""
    echo "Options:"
    echo "  --follow, -f        - Follow log output (like tail -f)"
    echo "  --grep PATTERN      - Filter logs with grep pattern"
    echo "  --tail N            - Show last N lines (default: 100)"
    echo "  --background, -b    - Run in background and save to file"
    echo "  --all              - Monitor all services"
    echo ""
    echo "Examples:"
    echo "  $0 bitcoin-forecast-app                  # Stream bitcoin-forecast-app logs"
    echo "  $0 data-collector                       # Stream data-collector logs"
    echo "  $0 bitcoin-forecast-app --grep 'Made prediction for timestamp' # Filter for predictions"
    echo "  $0 data-collector --grep 'Saved data to'   # Filter for data saves"
    echo "  $0 web-app --background                  # Background monitoring"
    echo "  $0 --all --tail 50                      # Monitor all services"
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
GREP_PATTERN=""
TAIL="100"
BACKGROUND=""
SHOW_ALL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--follow)
            FOLLOW="-f"
            shift
            ;;
        --grep)
            GREP_PATTERN="$2"
            shift 2
            ;;
        -t|--tail)
            TAIL="$2"
            shift 2
            ;;
        -b|--background)
            BACKGROUND="true"
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
        bitcoin-forecast-app|data-collector|web-app|dashboard|kafka|kafka-ui|zookeeper)
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

# Function to monitor single service
monitor_service() {
    local service=$1
    local follow=$2
    local grep_pattern=$3
    local tail=$4
    local background=$5
    
    if ! kubectl get deployment $service -n bitcoin-prediction &> /dev/null; then
        print_error "Service $service not found"
        return 1
    fi
    
    # Always use -f for streaming logs when not in background mode
    if [[ "$background" != "true" ]] && [[ -z "$follow" ]]; then
        follow="-f"
    fi
    
    local cmd="kubectl logs deployment/$service -n bitcoin-prediction --tail=$tail $follow"
    
    if [[ -n "$grep_pattern" ]]; then
        cmd="$cmd | grep --line-buffered '$grep_pattern'"
    fi
    
    if [[ "$background" == "true" ]]; then
        local logfile="logs/${service}_$(date +%Y%m%d_%H%M%S).log"
        mkdir -p logs
        print_info "Starting background monitoring for $service"
        print_info "Logs will be saved to: $logfile"
        print_warning "Use 'tail -f $logfile' to view logs"
        print_warning "Use 'pkill -f \"kubectl logs.*$service\"' to stop"
        
        eval "$cmd" > "$logfile" 2>&1 &
        local pid=$!
        echo "$pid" > "logs/${service}.pid"
        print_status "Background monitoring started (PID: $pid)"
    else
        if [[ -n "$grep_pattern" ]]; then
            print_info "Monitoring $service for pattern: '$grep_pattern' (streaming)"
        else
            print_info "Monitoring $service (streaming logs, last $tail lines)"
        fi
        print_warning "Press Ctrl+C to stop monitoring"
        echo "----------------------------------------"
        eval "$cmd"
    fi
}

# Function to monitor all services
monitor_all_services() {
    local services=("bitcoin-forecast-app" "data-collector" "web-app" "dashboard" "kafka" "kafka-ui" "zookeeper")
    
    print_info "Starting monitoring for all services..."
    mkdir -p logs
    
    for service in "${services[@]}"; do
        if kubectl get deployment $service -n bitcoin-prediction &> /dev/null; then
            local logfile="logs/${service}_$(date +%Y%m%d_%H%M%S).log"
            kubectl logs deployment/$service -n bitcoin-prediction --tail=$TAIL $FOLLOW > "$logfile" 2>&1 &
            local pid=$!
            echo "$pid" > "logs/${service}.pid"
            print_status "$service monitoring started (PID: $pid, Log: $logfile)"
        else
            print_warning "$service not found"
        fi
    done
    
    echo ""
    print_info "📋 Monitoring commands:"
    echo "View logs:     tail -f logs/<service>_*.log"
    echo "Stop all:      ./k8s/monitor.sh --stop-all"
    echo "List active:   ps aux | grep 'kubectl logs'"
}

# Function to stop all background monitoring
stop_all_monitoring() {
    print_info "Stopping all background monitoring..."
    
    if [[ -d "logs" ]]; then
        for pidfile in logs/*.pid; do
            if [[ -f "$pidfile" ]]; then
                local pid=$(cat "$pidfile")
                local service=$(basename "$pidfile" .pid)
                if kill "$pid" 2>/dev/null; then
                    print_status "Stopped monitoring for $service (PID: $pid)"
                else
                    print_warning "Could not stop PID $pid for $service (may have already stopped)"
                fi
                rm -f "$pidfile"
            fi
        done
    fi
    
    # Also kill any remaining kubectl logs processes
    pkill -f "kubectl logs.*bitcoin-prediction" 2>/dev/null || true
    print_status "All monitoring stopped"
}

# Special commands
if [[ "$1" == "--stop-all" ]]; then
    stop_all_monitoring
    exit 0
fi

# No special pattern handling needed anymore

# Main logic
if [[ "$SHOW_ALL" == "true" ]]; then
    monitor_all_services
elif [[ -z "$SERVICE" ]]; then
    print_error "Please specify a service name or use --all"
    echo ""
    show_usage
    exit 1
else
    monitor_service "$SERVICE" "$FOLLOW" "$GREP_PATTERN" "$TAIL" "$BACKGROUND"
fi 