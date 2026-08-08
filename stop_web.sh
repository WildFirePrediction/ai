#!/bin/bash

# Stop Script for the Interactive Web Demo
# Terminates the web server together with the inference engine it hosts.
#
# Usage:
#   ./stop_web.sh              # stop the demo started by ./start_web.sh
#   ./stop_web.sh --port 9000  # also free a non-default port

set -u

cd "$(dirname "$0")"

PORT=8080
PID_FILE="webdemo/.webdemo.pid"

while [ $# -gt 0 ]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        -h|--help)
            sed -n '3,9p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "ERROR: Unknown option: $1"
            exit 1 ;;
    esac
done

echo "========================================"
echo "Stopping Wildfire Web Demo"
echo "========================================"

STOPPED=0

# Graceful shutdown, escalating to SIGKILL if the process ignores SIGTERM
stop_pid() {
    local pid="$1"
    local label="$2"

    if ! kill -0 "$pid" 2>/dev/null; then
        return 1
    fi

    echo "  Stopping $label (PID $pid)..."
    kill "$pid" 2>/dev/null

    for _ in $(seq 1 10); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "  Stopped"
            return 0
        fi
        sleep 1
    done

    echo "  Did not exit, sending SIGKILL"
    kill -9 "$pid" 2>/dev/null
    sleep 1
    return 0
}

# 1) PID file written by start_web.sh
if [ -f "$PID_FILE" ]; then
    PID="$(cat "$PID_FILE")"
    if stop_pid "$PID" "web demo server"; then
        STOPPED=1
    else
        echo "  Stale PID file (process $PID not running)"
    fi
    rm -f "$PID_FILE"
fi

# 2) Any server process started without the PID file (e.g. foreground mode)
LEFTOVERS="$(pgrep -f "webdemo/server.py" 2>/dev/null)"
if [ -n "$LEFTOVERS" ]; then
    for PID in $LEFTOVERS; do
        if stop_pid "$PID" "leftover server process"; then
            STOPPED=1
        fi
    done
fi

# 3) Anything still holding the demo port
if command -v lsof >/dev/null 2>&1; then
    PORT_PIDS="$(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null)"
    if [ -n "$PORT_PIDS" ]; then
        for PID in $PORT_PIDS; do
            if stop_pid "$PID" "process on port $PORT"; then
                STOPPED=1
            fi
        done
    fi
fi

echo ""
if [ "$STOPPED" -eq 1 ]; then
    echo "Web demo and inference engine stopped"
else
    echo "Nothing to stop (web demo was not running)"
fi

# Report remaining GPU usage so a stuck CUDA context is visible
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)"
    echo "GPU memory in use: $USED"
fi
echo ""
