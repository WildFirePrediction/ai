#!/bin/bash
# A3C V8 LSTM RECALL SMALL Training Script - RECALL-FIRST SUPERAGGRO
# SMALL MODEL: 3.5x smaller to combat overfitting

cd "$(dirname "$0")"

# Configuration
NUM_WORKERS=8         # Balanced (between V6's 4 and V7's 12)
MAX_EPISODES=15000    # Balanced
LR=3e-4              # Same as V6
GAMMA=0.99
SEQUENCE_LENGTH=3
MEL_THRESHOLD=4
SEED=42

DATA_DIR="/home/chaseungjoon/code/WildfirePrediction/embedded_data/fire_episodes_16ch_normalized"
CHECKPOINT_DIR="./checkpoints"

echo "==========================================================="
echo "A3C V8 LSTM RECALL SMALL - RECALL-FIRST SUPERAGGRO"
echo "==========================================================="
echo "SMALL MODEL to Combat Overfitting:"
echo "  - ~500K parameters (3.5x smaller than V6)"
echo "  - 128 LSTM hidden (was 256 in V6)"
echo "  - Narrower CNN: 32→64→128 (was 64→128→256)"
echo "  - GroupNorm (proven in V6)"
echo "  - 8 workers (balanced)"
echo "  - 15k episodes"
echo ""
echo "Goal: Better generalization"
echo "  V6 train/val gap: 3.16x (0.4356 vs 0.1379)"
echo "  V8 target gap: <2.0x"
echo ""
echo "RECALL-FIRST SUPERAGGRO Reward System:"
echo "  - Missing fire spread (FN): -500.0 penalty"
echo "  - False alarm (FP): +20.0 REWARD"
echo "  - Correct prediction with 95%+ recall: +200.0"
echo "  - Correct prediction with 80%+ recall: +100.0"
echo "  - Correct prediction with 50%+ recall: +50.0"
echo "  - Correct prediction with <50% recall: -100.0"
echo "  - Small precision penalty: -(1-p)*2.0"
echo "  - Correct silence: 0.0"
echo ""
echo "Philosophy: Coverage is EVERYTHING."
echo "            Better to evacuate an entire mountain"
echo "            than to miss one house."
echo "==========================================================="
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
echo "==========================================================="

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

echo "==========================================================="
echo "SMALL MODEL Training completed!"
echo "==========================================================="
