#!/bin/bash
# A3C V7 LSTM RECALL (LARGE) Training Script - RECALL-FIRST SUPERAGGRO
# LARGE MODEL: 3.4x bigger, 12 workers, 25k episodes, LR schedule

cd "$(dirname "$0")"

# Configuration
NUM_WORKERS=12        # 3x more (was 4)
MAX_EPISODES=25000    # 2.5x more (was 10k)
LR=5e-4              # Higher initial (was 3e-4)
MIN_LR=1e-5          # Cosine annealing minimum
GAMMA=0.99
SEQUENCE_LENGTH=3
MEL_THRESHOLD=4
SEED=42

DATA_DIR="/home/chaseungjoon/code/WildfirePrediction/embedded_data/fire_episodes_16ch_normalized"
CHECKPOINT_DIR="./checkpoints"

echo "==========================================================="
echo "A3C V7 LSTM RECALL (LARGE) - RECALL-FIRST SUPERAGGRO"
echo "==========================================================="
echo "LARGE MODEL IMPROVEMENTS:"
echo "  - ~6M parameters (3.4x larger than V6)"
echo "  - 512 LSTM hidden (was 256)"
echo "  - 4-layer CNN: 64→128→256→512 (was 3-layer)"
echo "  - BatchNorm for better gradient flow"
echo "  - Larger policy head: 512 units (was 256)"
echo "  - 12 workers (was 4) - 3x parallelism"
echo "  - 25k episodes (was 10k)"
echo "  - Learning rate schedule: $LR → $MIN_LR (cosine)"
echo "  - Higher entropy coef: 0.02 (more exploration)"
echo "  - Higher grad clip: 1.0 (larger updates)"
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
echo "  Initial learning rate: $LR"
echo "  Min learning rate: $MIN_LR"
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
    --min-lr $MIN_LR \
    --gamma $GAMMA \
    --sequence-length $SEQUENCE_LENGTH \
    --mel-threshold $MEL_THRESHOLD \
    --seed $SEED

echo "==========================================================="
echo "LARGE MODEL Training completed!"
echo "==========================================================="
