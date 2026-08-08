#!/bin/bash

# Start Script for the Interactive Web Demo
# Loads the A3C inference engine and serves the map UI on all interfaces so the
# demo is reachable over Tailscale.
#
# Usage:
#   ./start_web.sh                 # start in background on port 8080
#   ./start_web.sh --port 9000     # custom port
#   ./start_web.sh --device cpu    # force CPU inference
#   ./start_web.sh --foreground    # run in this terminal (Ctrl+C to stop)

set -u

cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"

PORT=8080
DEVICE=""
FOREGROUND=0
CHECKPOINT="rl_training/a3c_16ch/V3_LSTM_REL/checkpoints/run1_relaxed/best_model.pt"
DATA_DIR="embedded_data"

LOG_DIR="webdemo/logs"
LOG_FILE="$LOG_DIR/webdemo.log"
PID_FILE="webdemo/.webdemo.pid"

# ----------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --port)       PORT="$2"; shift 2 ;;
        --device)     DEVICE="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --data-dir)   DATA_DIR="$2"; shift 2 ;;
        --foreground) FOREGROUND=1; shift ;;
        -h|--help)
            sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "ERROR: Unknown option: $1"
            echo "Run './start_web.sh --help' for usage"
            exit 1 ;;
    esac
done

echo "========================================"
echo "Wildfire Prediction - Web Demo"
echo "========================================"

# ----------------------------------------------------------------------
# Preflight checks
# ----------------------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "ERROR: Virtual environment not found at .venv/"
    echo "Create it first: python3 -m venv .venv && pip install -r requirements.txt"
    exit 1
fi

source .venv/bin/activate

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Model checkpoint not found: $CHECKPOINT"
    exit 1
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Environmental data not found: $DATA_DIR/"
    echo "Download it first: ./download_data.sh"
    exit 1
fi

# Refuse to start twice on the same port
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "ERROR: Web demo already running (PID $(cat "$PID_FILE"))"
    echo "Stop it first: ./stop_web.sh"
    exit 1
fi

if command -v ss >/dev/null 2>&1 && ss -tln 2>/dev/null | grep -q ":$PORT "; then
    echo "ERROR: Port $PORT is already in use"
    echo "Pick another port: ./start_web.sh --port 9000"
    exit 1
fi

# ----------------------------------------------------------------------
# Device selection
# ----------------------------------------------------------------------
if [ -z "$DEVICE" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        DEVICE="cuda"
        echo "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    else
        DEVICE="cpu"
        echo "WARNING: No GPU detected, running on CPU"
    fi
fi

mkdir -p "$LOG_DIR" webdemo/outputs

SERVER_ARGS="--host 0.0.0.0 --port $PORT --device $DEVICE"
SERVER_ARGS="$SERVER_ARGS --checkpoint $CHECKPOINT --data-dir $DATA_DIR"

# ----------------------------------------------------------------------
# Foreground mode
# ----------------------------------------------------------------------
if [ "$FOREGROUND" -eq 1 ]; then
    echo "Starting web demo in foreground (Ctrl+C to stop)"
    echo ""
    exec python webdemo/server.py $SERVER_ARGS
fi

# ----------------------------------------------------------------------
# Background mode
# ----------------------------------------------------------------------
echo "Starting inference engine and web server on port $PORT..."
nohup python webdemo/server.py $SERVER_ARGS > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

# Wait for the model to load and the health endpoint to answer
echo -n "Loading model"
READY=0
for _ in $(seq 1 60); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "ERROR: Server process exited during startup. Last log lines:"
        echo "----------------------------------------"
        tail -25 "$LOG_FILE"
        echo "----------------------------------------"
        rm -f "$PID_FILE"
        exit 1
    fi

    if curl -sf "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
        READY=1
        break
    fi

    echo -n "."
    sleep 1
done
echo ""

if [ "$READY" -ne 1 ]; then
    echo "ERROR: Server did not become ready within 60s. Last log lines:"
    echo "----------------------------------------"
    tail -25 "$LOG_FILE"
    echo "----------------------------------------"
    kill "$SERVER_PID" 2>/dev/null
    rm -f "$PID_FILE"
    exit 1
fi

# ----------------------------------------------------------------------
# Access URLs
# ----------------------------------------------------------------------
TAILSCALE_IP=""
if command -v tailscale >/dev/null 2>&1; then
    TAILSCALE_IP="$(tailscale ip -4 2>/dev/null | head -1)"
    if [ -z "$TAILSCALE_IP" ]; then
        echo "NOTE: Tailscale is installed but not connected."
        echo "      Bring it up with 'sudo tailscale up' for remote access."
    fi
fi

LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"

echo ""
echo "========================================"
echo "Web demo is running"
echo "========================================"
echo "  PID:      $SERVER_PID"
echo "  Device:   $DEVICE"
echo "  Log:      $PROJECT_ROOT/$LOG_FILE"
echo "  Outputs:  $PROJECT_ROOT/webdemo/outputs/"
echo ""
echo "Open in a browser:"
if [ -n "$TAILSCALE_IP" ]; then
    echo "  Tailscale:  http://$TAILSCALE_IP:$PORT"
fi
if [ -n "$LAN_IP" ]; then
    echo "  LAN:        http://$LAN_IP:$PORT"
fi
echo "  Local:      http://localhost:$PORT"
echo ""
echo "Stop everything with: ./stop_web.sh"
echo "Follow the log with:  tail -f $LOG_FILE"
echo ""
