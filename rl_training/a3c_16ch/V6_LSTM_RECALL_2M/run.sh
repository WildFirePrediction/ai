#!/bin/bash
# A3C V6 LSTM RECALL Training Script - RECALL-FIRST SUPERAGGRO Reward Shaping
# Recall-first reward system: -500 for missing spread, +20 for false alarm, +200 for 95%+ recall

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
echo "A3C V6 LSTM RECALL - RECALL-FIRST SUPERAGGRO"
echo "=============================================="
echo "RECALL-FIRST SUPERAGGRO Reward System:"
echo "  - Missing fire spread (false negative): -500.0 penalty (CATASTROPHIC - 5x worse!)"
echo "  - False alarm (false positive): +20.0 REWARD (4x better - encourage caution!)"
echo "  - Correct prediction with 95%+ recall: +200.0 (MASSIVE)"
echo "  - Correct prediction with 80%+ recall: +100.0 (Great)"
echo "  - Correct prediction with 50%+ recall: +50.0 (OK)"
echo "  - Correct prediction with <50% recall: -100.0 (Bad)"
echo "  - Small precision penalty: -(1-precision)*2.0"
echo "  - Correct silence: 0.0 (neutral)"
echo ""
echo "Philosophy: Coverage is EVERYTHING."
echo "            Better to evacuate an entire mountain"
echo "            than to miss one house."
echo "            25:1 risk ratio massively biases toward over-prediction."
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
