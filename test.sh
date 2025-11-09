#!/usr/bin/env bash
# Test a trained model on the TEST split
# ⚠️  WARNING: Only run this ONCE with your final model!
#
# Usage:
#   ./test.sh model_000100.pt              # Test specific checkpoint
#   ./test.sh                              # Find and use latest checkpoint

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Warning message
echo ""
echo "╔═════════════════════════════════════════════════════════════╗"
echo "║                       ⚠️  WARNING ⚠️                         ║"
echo "╟─────────────────────────────────────────────────────────────╢"
echo "║  You are about to evaluate on the TEST split!              ║"
echo "║                                                             ║"
echo "║  The test split should ONLY be used ONCE with your         ║"
echo "║  final model to get unbiased performance metrics.          ║"
echo "║                                                             ║"
echo "║  For model development and hyperparameter tuning,          ║"
echo "║  use the VALIDATION split instead: ./validate.sh           ║"
echo "╚═════════════════════════════════════════════════════════════╝"
echo ""

# Confirm with user
read -p "Are you sure you want to evaluate on TEST split? (yes/no): " -r
echo
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "Test evaluation cancelled."
    echo "Use ./validate.sh for validation split evaluation instead."
    exit 0
fi

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
echo "TEST - Evaluating on Test Split (Final Evaluation)"
echo "================================================================"
echo ""

python3 -m rl_training.evaluate \
    --repo-root "$REPO_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --split test \
    --device cuda

echo ""
echo "================================================================"
echo "Test evaluation complete!"
echo "Results saved to: rl_training/results/"
echo ""
echo "⚠️  Remember: Do NOT use test results for model selection!"
echo "   Test results are for final reporting only."
echo "================================================================"
