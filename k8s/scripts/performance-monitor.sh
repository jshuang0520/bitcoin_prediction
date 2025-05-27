#!/bin/bash

# Bitcoin Prediction System - Performance Monitor
# Tracks real-time metrics to validate performance improvements

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

print_header() {
    echo ""
    echo -e "${CYAN}=== $1 ===${NC}"
    echo ""
}

print_metric() {
    local name=$1
    local value=$2
    local target=$3
    local unit=$4
    
    if [[ "$value" =~ ^[0-9]+\.?[0-9]*$ ]]; then
        if (( $(echo "$value <= $target" | bc -l) )); then
            echo -e "  ${GREEN}✅ $name: $value$unit (target: ≤$target$unit)${NC}"
        else
            echo -e "  ${YELLOW}⚠ $name: $value$unit (target: ≤$target$unit)${NC}"
        fi
    else
        echo -e "  ${BLUE}ℹ $name: $value$unit${NC}"
    fi
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

print_header "🚀 Bitcoin Prediction System - Performance Monitor"

# Function to get pod resource usage
get_pod_metrics() {
    local service=$1
    kubectl top pod -n bitcoin-prediction -l app=$service --no-headers 2>/dev/null | head -1
}

# Function to get pod status
get_pod_status() {
    local service=$1
    kubectl get pods -n bitcoin-prediction -l app=$service --no-headers 2>/dev/null | awk '{print $3}' | head -1
}

# Function to get pod count
get_pod_count() {
    local service=$1
    kubectl get pods -n bitcoin-prediction -l app=$service --no-headers 2>/dev/null | wc -l
}

# Function to get HPA status
get_hpa_status() {
    local service=$1
    kubectl get hpa ${service}-hpa -n bitcoin-prediction --no-headers 2>/dev/null | awk '{print $3 "/" $4 " replicas, " $5 " CPU, " $6 " Memory"}'
}

print_header "📊 Real-Time Performance Metrics"

# Service performance tracking
services=("data-collector" "bitcoin-forecast-app" "kafka" "zookeeper" "web-app" "dashboard")

for service in "${services[@]}"; do
    echo -e "${CYAN}🔍 $service:${NC}"
    
    # Pod status
    status=$(get_pod_status $service)
    pod_count=$(get_pod_count $service)
    
    if [ "$status" = "Running" ]; then
        print_status "Status: Running ($pod_count pods)"
    else
        print_warning "Status: $status ($pod_count pods)"
    fi
    
    # Resource usage
    metrics=$(get_pod_metrics $service)
    if [ -n "$metrics" ]; then
        cpu=$(echo $metrics | awk '{print $2}' | sed 's/m//')
        memory=$(echo $metrics | awk '{print $3}' | sed 's/Mi//')
        
        # Performance targets based on service type
        case $service in
            "data-collector")
                print_metric "CPU Usage" "$cpu" "800" "m"
                print_metric "Memory Usage" "$memory" "800" "Mi"
                ;;
            "bitcoin-forecast-app")
                print_metric "CPU Usage" "$cpu" "3000" "m"
                print_metric "Memory Usage" "$memory" "6000" "Mi"
                ;;
            "kafka")
                print_metric "CPU Usage" "$cpu" "1500" "m"
                print_metric "Memory Usage" "$memory" "3000" "Mi"
                ;;
            "zookeeper")
                print_metric "CPU Usage" "$cpu" "400" "m"
                print_metric "Memory Usage" "$memory" "800" "Mi"
                ;;
            *)
                print_metric "CPU Usage" "$cpu" "500" "m"
                print_metric "Memory Usage" "$memory" "500" "Mi"
                ;;
        esac
    else
        print_warning "Metrics not available (pod may be starting)"
    fi
    
    # HPA status
    hpa_status=$(get_hpa_status $service)
    if [ -n "$hpa_status" ]; then
        echo -e "  ${BLUE}ℹ Auto-scaling: $hpa_status${NC}"
    fi
    
    echo ""
done

print_header "⚡ Real-Time Processing Performance"

# Check data collection rate
print_info "Checking data collection frequency..."
if kubectl logs deployment/data-collector -n bitcoin-prediction --tail=10 2>/dev/null | grep -q "Saved data"; then
    recent_logs=$(kubectl logs deployment/data-collector -n bitcoin-prediction --tail=20 --since=60s 2>/dev/null | grep "Saved data" | wc -l)
    if [ "$recent_logs" -gt 0 ]; then
        rate=$(echo "scale=2; $recent_logs / 60" | bc -l)
        print_metric "Data Collection Rate" "$rate" "1.0" " Hz"
    else
        print_warning "No recent data collection logs found"
    fi
else
    print_warning "Data collector logs not available"
fi

# Check prediction frequency
print_info "Checking prediction frequency..."
if kubectl logs deployment/bitcoin-forecast-app -n bitcoin-prediction --tail=10 2>/dev/null | grep -q "prediction"; then
    recent_predictions=$(kubectl logs deployment/bitcoin-forecast-app -n bitcoin-prediction --tail=20 --since=60s 2>/dev/null | grep -i "prediction" | wc -l)
    if [ "$recent_predictions" -gt 0 ]; then
        pred_rate=$(echo "scale=2; $recent_predictions / 60" | bc -l)
        print_metric "Prediction Rate" "$pred_rate" "1.0" " Hz"
    else
        print_warning "No recent prediction logs found"
    fi
else
    print_warning "Prediction logs not available"
fi

print_header "🎯 System Health & Availability"

# Check service endpoints
services_with_ports=("web-app:5000" "dashboard:8501" "kafka-ui:8080")

for service_port in "${services_with_ports[@]}"; do
    service=$(echo $service_port | cut -d: -f1)
    port=$(echo $service_port | cut -d: -f2)
    
    if kubectl get service $service -n bitcoin-prediction &> /dev/null; then
        print_status "$service service is available"
    else
        print_error "$service service not found"
    fi
done

# Check persistent volumes
print_info "Checking data persistence..."
pvcs=$(kubectl get pvc -n bitcoin-prediction --no-headers 2>/dev/null | wc -l)
if [ "$pvcs" -gt 0 ]; then
    print_status "$pvcs persistent volume claims active"
    kubectl get pvc -n bitcoin-prediction --no-headers 2>/dev/null | while read line; do
        name=$(echo $line | awk '{print $1}')
        status=$(echo $line | awk '{print $2}')
        size=$(echo $line | awk '{print $4}')
        echo -e "  ${GREEN}✓ $name: $status ($size)${NC}"
    done
else
    print_warning "No persistent volumes found"
fi

print_header "📈 Performance Improvements Summary"

echo -e "${CYAN}🔧 Resource Enhancements:${NC}"
echo -e "  ${GREEN}✅ Data-Collector: +400% CPU, +300% Memory${NC}"
echo -e "  ${GREEN}✅ Bitcoin-Forecast-App: +100% CPU, +300% Memory${NC}"
echo -e "  ${GREEN}✅ Kafka: +300% CPU, +300% Memory${NC}"
echo -e "  ${GREEN}✅ ZooKeeper: +150% CPU, +100% Memory${NC}"

echo ""
echo -e "${CYAN}🎯 Priority Classes:${NC}"
echo -e "  ${RED}🔴 Critical: Bitcoin-Forecast-App, Kafka${NC}"
echo -e "  ${YELLOW}🟡 High: Data-Collector, ZooKeeper${NC}"
echo -e "  ${GREEN}🟢 Normal: Web-App, Dashboard, Kafka-UI${NC}"

echo ""
echo -e "${CYAN}🔄 Auto-Scaling:${NC}"
echo -e "  ${GREEN}✅ Faster scale-up (30-60s) for real-time processing${NC}"
echo -e "  ${GREEN}✅ Lower CPU thresholds (60-70%) for responsive scaling${NC}"
echo -e "  ${GREEN}✅ Pod anti-affinity for ML workload distribution${NC}"

echo ""
echo -e "${CYAN}⚡ Expected Performance:${NC}"
echo -e "  ${GREEN}✅ Sub-second data collection latency${NC}"
echo -e "  ${GREEN}✅ <1s prediction cycle time${NC}"
echo -e "  ${GREEN}✅ Real-time dashboard updates${NC}"
echo -e "  ${GREEN}✅ High-throughput Kafka streaming${NC}"

print_header "🔧 Monitoring Commands"

echo "📋 Useful commands for ongoing monitoring:"
echo ""
echo "  # Watch resource usage in real-time:"
echo "  watch kubectl top pods -n bitcoin-prediction"
echo ""
echo "  # Monitor auto-scaling:"
echo "  watch kubectl get hpa -n bitcoin-prediction"
echo ""
echo "  # Check service logs:"
echo "  kubectl logs -f deployment/data-collector -n bitcoin-prediction"
echo "  kubectl logs -f deployment/bitcoin-forecast-app -n bitcoin-prediction"
echo ""
echo "  # Performance monitoring:"
echo "  ./k8s/scripts/performance-monitor.sh"

print_status "Performance monitoring complete!"
print_info "💡 Run this script periodically to track system performance" 