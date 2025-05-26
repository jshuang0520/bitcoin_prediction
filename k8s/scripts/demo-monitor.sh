#!/bin/bash

# Bitcoin Prediction System - Demo Monitoring Script
# Shows only clean logs for demonstration purposes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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
    echo "Usage: $0 [service] [options]"
    echo ""
    echo "Demo Services (clean logs only):"
    echo "  predictions     - Show only successful Bitcoin predictions"
    echo "  data-saves      - Show only successful data saves"
    echo "  both           - Show both predictions and data saves"
    echo ""
    echo "Options:"
    echo "  --background, -b    - Run in background and save to file"
    echo "  --tail N           - Show last N lines (default: 50)"
    echo ""
    echo "Examples:"
    echo "  $0 predictions                    # Stream prediction logs"
    echo "  $0 data-saves                    # Stream data save logs"
    echo "  $0 both                          # Stream both types"
    echo "  $0 predictions --background      # Background monitoring"
}

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    print_error "kubectl is not installed or not in PATH"
    exit 1
fi

# Check if minikube is running
if ! minikube status &> /dev/null; then
    print_error "minikube is not running"
    echo "Start with: ./k8s/scripts/start-minikube.sh"
    exit 1
fi

# Parse arguments
SERVICE=""
BACKGROUND=""
TAIL="50"

while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--background)
            BACKGROUND="true"
            shift
            ;;
        -t|--tail)
            TAIL="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        predictions|data-saves|both)
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

# Function to monitor predictions (clean logs only)
monitor_predictions() {
    local background=$1
    
    if ! kubectl get deployment bitcoin-forecast-app -n bitcoin-prediction &> /dev/null; then
        print_error "bitcoin-forecast-app not found"
        return 1
    fi
    
    local cmd="kubectl logs deployment/bitcoin-forecast-app -n bitcoin-prediction --tail=$TAIL -f | grep --line-buffered 'Made prediction for timestamp'"
    
    if [[ "$background" == "true" ]]; then
        local logfile="logs/demo_predictions_$(date +%Y%m%d_%H%M%S).log"
        mkdir -p logs
        print_info "Starting background monitoring for Bitcoin predictions"
        print_info "Clean prediction logs will be saved to: $logfile"
        print_warning "Use 'tail -f $logfile' to view logs"
        print_warning "Use 'pkill -f \"kubectl logs.*bitcoin-forecast-app\"' to stop"
        
        eval "$cmd" > "$logfile" 2>&1 &
        local pid=$!
        echo "$pid" > "logs/demo_predictions.pid"
        print_status "Background prediction monitoring started (PID: $pid)"
    else
        print_info "🔮 Monitoring Bitcoin Predictions (demo mode - clean logs only)"
        print_warning "Press Ctrl+C to stop monitoring"
        echo "----------------------------------------"
        eval "$cmd"
    fi
}

# Function to monitor data saves (clean logs only)
monitor_data_saves() {
    local background=$1
    
    if ! kubectl get deployment data-collector -n bitcoin-prediction &> /dev/null; then
        print_error "data-collector not found"
        return 1
    fi
    
    local cmd="kubectl logs deployment/data-collector -n bitcoin-prediction --tail=$TAIL -f | grep --line-buffered 'Saved data to'"
    
    if [[ "$background" == "true" ]]; then
        local logfile="logs/demo_data_saves_$(date +%Y%m%d_%H%M%S).log"
        mkdir -p logs
        print_info "Starting background monitoring for data saves"
        print_info "Clean data save logs will be saved to: $logfile"
        print_warning "Use 'tail -f $logfile' to view logs"
        print_warning "Use 'pkill -f \"kubectl logs.*data-collector\"' to stop"
        
        eval "$cmd" > "$logfile" 2>&1 &
        local pid=$!
        echo "$pid" > "logs/demo_data_saves.pid"
        print_status "Background data save monitoring started (PID: $pid)"
    else
        print_info "💾 Monitoring Data Collection (demo mode - clean logs only)"
        print_warning "Press Ctrl+C to stop monitoring"
        echo "----------------------------------------"
        eval "$cmd"
    fi
}

# Function to monitor both services
monitor_both() {
    local background=$1
    
    if [[ "$background" == "true" ]]; then
        print_info "Starting background monitoring for both services..."
        monitor_predictions "true"
        monitor_data_saves "true"
        echo ""
        print_info "📋 Demo monitoring commands:"
        echo "View predictions: tail -f logs/demo_predictions_*.log"
        echo "View data saves:  tail -f logs/demo_data_saves_*.log"
        echo "Stop all:         ./k8s/scripts/demo-monitor.sh --stop-all"
    else
        print_info "🎬 Demo Mode: Monitoring Both Services (clean logs only)"
        print_warning "Press Ctrl+C to stop monitoring"
        echo "========================================"
        
        # Run both in parallel with different colors
        (
            echo -e "${CYAN}[PREDICTIONS]${NC}"
            kubectl logs deployment/bitcoin-forecast-app -n bitcoin-prediction --tail=$TAIL -f | \
            grep --line-buffered 'Made prediction for timestamp' | \
            sed "s/^/$(echo -e "${CYAN}[PRED]${NC}") /"
        ) &
        
        (
            echo -e "${GREEN}[DATA SAVES]${NC}"
            kubectl logs deployment/data-collector -n bitcoin-prediction --tail=$TAIL -f | \
            grep --line-buffered 'Saved data to' | \
            sed "s/^/$(echo -e "${GREEN}[DATA]${NC}") /"
        ) &
        
        # Wait for both background processes
        wait
    fi
}

# Function to stop all background monitoring
stop_all_monitoring() {
    print_info "Stopping all demo background monitoring..."
    
    if [[ -d "logs" ]]; then
        for pidfile in logs/demo_*.pid; do
            if [[ -f "$pidfile" ]]; then
                local pid=$(cat "$pidfile")
                local service=$(basename "$pidfile" .pid)
                if kill "$pid" 2>/dev/null; then
                    print_status "Stopped demo monitoring for $service (PID: $pid)"
                else
                    print_warning "Could not stop PID $pid for $service (may have already stopped)"
                fi
                rm -f "$pidfile"
            fi
        done
    fi
    
    # Also kill any remaining kubectl logs processes for demo
    pkill -f "kubectl logs.*bitcoin-prediction.*grep" 2>/dev/null || true
    print_status "All demo monitoring stopped"
}

# Special commands
if [[ "$1" == "--stop-all" ]]; then
    stop_all_monitoring
    exit 0
fi

# Main logic
if [[ -z "$SERVICE" ]]; then
    print_error "Please specify a service: predictions, data-saves, or both"
    echo ""
    show_usage
    exit 1
fi

case "$SERVICE" in
    predictions)
        monitor_predictions "$BACKGROUND"
        ;;
    data-saves)
        monitor_data_saves "$BACKGROUND"
        ;;
    both)
        monitor_both "$BACKGROUND"
        ;;
    *)
        print_error "Unknown service: $SERVICE"
        show_usage
        exit 1
        ;;
esac 