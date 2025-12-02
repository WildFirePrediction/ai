#!/bin/bash

# A3C 16ch LSTM Training with Normalized Data
# Run from WildfirePrediction/rl_training/a3c_16ch/V3_LSTM directory

PYTHONPATH=/home/chaseungjoon/code/WildfirePrediction:$PYTHONPATH \
python3 train.py \
  --data-dir /home/chaseungjoon/code/WildfirePrediction/embedded_data/fire_episodes_16ch_normalized \
  --checkpoint-dir ./checkpoints/mel4-lstm-norm-10k \
  --mel-threshold 4 \
  --num-workers 4 \
  --max-episodes 10000 \
  --lr 3e-4 \
  --sequence-length 3 \
  --seed 42
