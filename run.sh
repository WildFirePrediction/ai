#!/usr/bin/env bash
# Simple script to run FULL training on all data
# Usage:
#   ./run.sh                    # Start fresh training
#   ./run.sh model_000100.pt    # Resume from checkpoint

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Check if resuming from checkpoint
if [[ $# -gt 0 ]]; then
    CHECKPOINT="$1"
    echo "Resuming from checkpoint: $CHECKPOINT"
    ./rl_training/run_training.sh \
        --max-iters 1000 \
        --batch-envs 4 \
        --save-every 10 \
        --split train \
        --resume "$CHECKPOINT"
else
    echo "Starting FULL training on ALL training data"
    ./rl_training/run_training.sh \
        --max-iters 1000 \
        --batch-envs 4 \
        --save-every 10 \
        --split train
fi
