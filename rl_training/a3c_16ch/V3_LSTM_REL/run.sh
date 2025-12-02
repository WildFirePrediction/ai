#!/bin/bash
# A3C V3 LSTM REL Training Script - Relaxed IoU (8-neighbor tolerance)

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

# Change to script directory
cd "$SCRIPT_DIR"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Configuration
DATA_DIR="$PROJECT_ROOT/embedded_data/fire_episodes_16ch_normalized"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints/run1_relaxed"
MEL_THRESHOLD=4
NUM_WORKERS=4
MAX_EPISODES=10000
LR=3e-4
GAMMA=0.99
SEQUENCE_LENGTH=3
SEED=42

echo "========================================================================"
echo "A3C V3 LSTM REL TRAINING - RELAXED IoU (8-neighbor tolerance)"
echo "========================================================================"
echo "Data: $DATA_DIR"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "MEL Threshold: $MEL_THRESHOLD"
echo "Workers: $NUM_WORKERS"
echo "Max Episodes: $MAX_EPISODES"
echo "Learning Rate: $LR"
echo "Gamma: $GAMMA"
echo "Sequence Length: $SEQUENCE_LENGTH"
echo "Seed: $SEED"
echo "Relaxed IoU: 3x3 dilation (8-neighbor tolerance)"
echo "========================================================================"
echo ""

# Run training
"$PROJECT_ROOT/.venv/bin/python3" train.py \
    --data-dir "$DATA_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --mel-threshold $MEL_THRESHOLD \
    --num-workers $NUM_WORKERS \
    --max-episodes $MAX_EPISODES \
    --lr $LR \
    --gamma $GAMMA \
    --sequence-length $SEQUENCE_LENGTH \
    --seed $SEED
