#!/bin/bash

echo "🧪 Testing Kubernetes Log Streaming"
echo "=================================="
echo ""

echo "📊 Current Pod Status:"
kubectl get pods -n bitcoin-prediction
echo ""

echo "🔍 Testing streaming logs for data-collector (5 seconds)..."
echo "You should see real-time Bitcoin price collection:"
echo "----------------------------------------"

# Test streaming for 5 seconds
timeout 5s kubectl logs deployment/data-collector -n bitcoin-prediction -f --tail=3 || true

echo ""
echo "----------------------------------------"
echo "✅ Streaming test completed!"
echo ""

echo "💡 To use the monitor script:"
echo "  ./k8s/monitor.sh data-collector                    # Stream data-collector logs"
echo "  ./k8s/monitor.sh bitcoin-forecast-app             # Stream bitcoin-forecast-app logs"
echo "  ./k8s/monitor.sh data-collector --grep 'Saved'    # Filter for data saves"
echo "  ./k8s/monitor.sh bitcoin-forecast-app --grep 'prediction' # Filter for predictions" 