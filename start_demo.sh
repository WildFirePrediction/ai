#!/bin/bash

# Quick Start Script for Demo Mode
# Starts Flask server and demo monitor

echo "========================================"
echo "Wildfire Demo System - Quick Start"
echo "========================================"

# Change to project directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found at .venv/"
    echo "Please create virtual environment first"
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    DEVICE="cuda"
else
    echo "WARNING: No GPU detected, using CPU"
    DEVICE="cpu"
fi

echo ""
echo "Starting DEMO MODE..."
echo ""

# Start Flask server in background
echo "[1/2] Starting Flask Inference Server on port 5000..."
python inference/rl/api_server.py --port 5000 --device $DEVICE &
FLASK_PID=$!

# Wait for Flask server to initialize
sleep 5

# Check if Flask server is running
if ! curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "ERROR: Flask server failed to start"
    kill $FLASK_PID 2>/dev/null
    exit 1
fi

echo "  Flask server is running (PID: $FLASK_PID)"
echo ""

# Start KFS monitor in DEMO mode
echo "[2/2] Starting Demo Fire Generator..."
echo "  Mode: DEMO (fake fires)"
echo "  Poll interval: 120 seconds (2 minutes)"
echo "  Output: inference/demo_rl/output/"
echo "  Press Ctrl+C to stop both services"
echo ""
echo "========================================"
echo ""

# Start monitor in DEMO mode (will run in foreground)
python inference/fire_monitor/kfs_monitor.py --demo --poll-interval 120

# Cleanup: Kill Flask server when monitor stops
echo ""
echo "Stopping Flask server..."
kill $FLASK_PID 2>/dev/null

echo "All services stopped"
echo ""
echo "Check outputs in: inference/demo_rl/output/"
