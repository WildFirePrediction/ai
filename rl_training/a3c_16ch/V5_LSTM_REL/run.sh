#!/bin/bash
# A3C V5 LSTM REL Training Script - AGGRESSIVE SAFETY-FIRST Reward Shaping
# Safety-first reward system: -100 for missing spread, +5 for false alarm (REWARD!), +50x IoU for correct

cd "$(dirname "$0")"

# Configuration
NUM_WORKERS=4
MAX_EPISODES=10000
LR=3e-4
GAMMA=0.99
SEQUENCE_LENGTH=3
MEL_THRESHOLD=4
SEED=42

DATA_DIR="/home/chaseungjoon/code/WildfirePrediction/embedded_data/fire_episodes_16ch_normalized"
CHECKPOINT_DIR="./checkpoints"

echo "=============================================="
echo "A3C V5 LSTM REL - AGGRESSIVE SAFETY-FIRST"
echo "=============================================="
echo "AGGRESSIVE SAFETY-FIRST Reward System:"
echo "  - Missing fire spread (false negative): -100.0 penalty (CATASTROPHIC!)"
echo "  - False alarm (false positive): +5.0 REWARD (Better safe than sorry!)"
echo "  - Correct prediction: relaxed_iou * 50.0 (VERY HIGH REWARD)"
echo "  - Correct silence: +1.0 reward (small)"
echo ""
echo "Philosophy: Better to warn 10 areas unnecessarily"
echo "            than to miss 1 area that will burn."
echo "=============================================="
echo "Configuration:"
echo "  Data directory: $DATA_DIR"
echo "  Checkpoint directory: $CHECKPOINT_DIR"
echo "  Number of workers: $NUM_WORKERS"
echo "  Max episodes: $MAX_EPISODES"
echo "  Learning rate: $LR"
echo "  Gamma: $GAMMA"
echo "  Sequence length: $SEQUENCE_LENGTH"
echo "  MEL threshold: $MEL_THRESHOLD"
echo "  Random seed: $SEED"
echo "=============================================="

mkdir -p "$CHECKPOINT_DIR"

# Run training
PYTHONPATH=/home/chaseungjoon/code/WildfirePrediction:$PYTHONPATH \
python3 train.py \
    --data-dir "$DATA_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --num-workers $NUM_WORKERS \
    --max-episodes $MAX_EPISODES \
    --lr $LR \
    --gamma $GAMMA \
    --sequence-length $SEQUENCE_LENGTH \
    --mel-threshold $MEL_THRESHOLD \
    --seed $SEED

echo "=============================================="
echo "Training completed!"
echo "=============================================="
