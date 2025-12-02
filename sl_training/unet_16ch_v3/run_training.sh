#!/bin/bash
# U-Net V3 Training Script - Dilated Ground Truth (8-neighbor tolerance)

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Change to script directory
cd "$SCRIPT_DIR"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Training configuration
DATA_DIR="$PROJECT_ROOT/embedded_data/fire_episodes_16ch_normalized"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints/run1_dilated"
BATCH_SIZE=32
EPOCHS=50
LR=1e-4
MIN_MEL=4
NUM_WORKERS=4
DEVICE="cuda"
WANDB_PROJECT="wildfire-unet-v3"
RUN_NAME="v3-dilated-8neighbor-$(date +%y%m%d-%H%M)"

echo "========================================================================"
echo "U-NET V3 TRAINING - DILATED GROUND TRUTH"
echo "========================================================================"
echo "Data: $DATA_DIR"
echo "Checkpoint: $CHECKPOINT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR"
echo "Device: $DEVICE"
echo "Wandb Project: $WANDB_PROJECT"
echo "Run Name: $RUN_NAME"
echo "========================================================================"
echo ""

# Run training
"$PROJECT_ROOT/.venv/bin/python3" train.py \
    --data-dir "$DATA_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --min-mel $MIN_MEL \
    --num-workers $NUM_WORKERS \
    --device $DEVICE \
    --wandb-project "$WANDB_PROJECT" \
    --run-name "$RUN_NAME"
