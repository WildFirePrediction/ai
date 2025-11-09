#!/usr/bin/env bash
# Validate a trained model on the validation split
# Usage:
#   ./validate.sh model_000100.pt              # Validate specific checkpoint
#   ./validate.sh                              # Find and use latest checkpoint

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Find checkpoint
if [[ $# -gt 0 ]]; then
    CHECKPOINT="$1"
    echo "Using specified checkpoint: $CHECKPOINT"
else
    # Find latest checkpoint
    CKPT_DIR="${REPO_ROOT}/rl_training/checkpoints"
    if [[ ! -d "$CKPT_DIR" ]] || [[ -z "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]]; then
        echo "ERROR: No checkpoints found in $CKPT_DIR"
        echo "Train a model first with: ./run.sh"
        exit 1
    fi
    CHECKPOINT=$(ls -t "$CKPT_DIR"/model_*.pt | head -1)
    echo "Using latest checkpoint: $(basename "$CHECKPOINT")"
fi

echo ""
echo "================================================================"
echo "VALIDATION - Evaluating on Validation Split"
echo "================================================================"
echo ""

python3 -m rl_training.evaluate \
    --repo-root "$REPO_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --split val \
    --device cuda

echo ""
echo "================================================================"
echo "Validation complete!"
echo "Results saved to: rl_training/results/"
echo "================================================================"
