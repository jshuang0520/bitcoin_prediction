#!/bin/bash

# Bitcoin Prediction System - Performance Monitor
# Real-time performance monitoring for optimized deployment

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${PURPLE}$1${NC}"
}

print_metric() {
    echo -e "${BLUE}$1${NC}"
}

print_status() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

# Check if system is running
if ! minikube status &> /dev/null; then
    print_error "❌ Minikube is not running"
    echo "Start with: ./k8s/startup.sh or ./k8s/build-performance-optimized.sh"
    exit 1
fi

if ! kubectl get namespace bitcoin-prediction &> /dev/null; then
    print_error "❌ Bitcoin prediction namespace not found"
    echo "Deploy with: ./k8s/build-from-scratch.sh or ./k8s/build-performance-optimized.sh"
    exit 1
fi

clear
print_header "🚀 Bitcoin Prediction System - Performance Monitor"
print_header "=================================================="

echo ""
print_metric "📊 Resource Utilization:"
kubectl top pods -n bitcoin-prediction 2>/dev/null || print_warning "Metrics not available yet (wait 1-2 minutes after startup)"

echo ""
print_metric "🔄 Auto-Scaling Status:"
kubectl get hpa -n bitcoin-prediction 2>/dev/null || print_warning "HPA not configured (use performance-optimized build)"

echo ""
print_metric "⚡ Service Performance:"
kubectl get pods -n bitcoin-prediction -o custom-columns="NAME:.metadata.name,STATUS:.status.phase,RESTARTS:.status.containerStatuses[*].restartCount,CPU_REQ:.spec.containers[*].resources.requests.cpu,MEM_REQ:.spec.containers[*].resources.requests.memory" 2>/dev/null

echo ""
print_metric "🎯 Performance Targets:"
echo "  Data-collector: <1s data collection frequency"
echo "  Bitcoin-forecast: <2s ML prediction time"
echo "  Web-app: <100ms API response time"
echo "  Overall: 99.9% uptime target"

echo ""
print_metric "🌐 Service URLs (Fixed):"
echo "  Web App:    http://localhost:5001"
echo "  Dashboard:  http://localhost:8501"
echo "  Kafka UI:   http://localhost:8080"

echo ""
print_metric "📈 Real-time Monitoring Commands:"
echo "  watch -n 5 ./k8s/monitor-performance.sh    # Auto-refresh every 5s"
echo "  ./k8s/logs.sh data-collector -f            # Follow data collection logs"
echo "  ./k8s/logs.sh bitcoin-forecast-app -f      # Follow ML prediction logs"

echo ""
print_metric "🔧 Performance Optimization:"
if kubectl get hpa -n bitcoin-prediction &> /dev/null; then
    print_status "✅ Auto-scaling enabled"
else
    print_warning "⚠️  Auto-scaling not enabled (use ./k8s/build-performance-optimized.sh)"
fi

if kubectl get priorityclass bitcoin-critical &> /dev/null; then
    print_status "✅ Priority classes configured"
else
    print_warning "⚠️  Priority classes not configured"
fi

# Check resource quotas
if kubectl get resourcequota -n bitcoin-prediction &> /dev/null; then
    print_status "✅ Resource quotas active"
else
    print_warning "⚠️  Resource quotas not configured"
fi

echo ""
print_metric "📊 System Health:"
ready_pods=$(kubectl get pods -n bitcoin-prediction --no-headers 2>/dev/null | grep -c "Running" || echo "0")
total_pods=$(kubectl get pods -n bitcoin-prediction --no-headers 2>/dev/null | wc -l || echo "0")

if [ "$ready_pods" -eq "$total_pods" ] && [ "$total_pods" -gt 0 ]; then
    print_status "✅ All $total_pods pods running"
else
    print_warning "⚠️  $ready_pods/$total_pods pods ready"
fi

echo ""
print_header "🎯 Performance Tips:"
echo "• Use 'kubectl top pods -n bitcoin-prediction' for real-time CPU/Memory"
echo "• Monitor HPA scaling with 'kubectl get hpa -n bitcoin-prediction -w'"
echo "• Check service logs with './k8s/logs.sh <service-name> -f'"
echo "• Test load balancing by refreshing http://localhost:5001 multiple times" 