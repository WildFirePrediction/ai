# A3C V6 - Phase 1 Improvements

## Overview

A3C V6 implements Phase 1 improvements from the roadmap:
1. **Data Augmentation** - Rotation (0°, 90°, 180°, 270°) and flipping (H/V)
2. **Experience Replay Buffer** - Stores top 100 high-IoU episodes
3. **Replay Sampling** - 30% of training samples from replay buffer

**Target Performance:** 45-50% IoU (from V3's 40.91%)

## Architecture

Same as V3:
- 3-layer CNN encoder (32→64→128 channels)
- Per-cell 8-neighbor prediction
- Dense rewards (IoU at every timestep)
- **417K parameters**

## Key Files

- `model_v6.py` - Model architecture (same as V3)
- `worker_v6.py` - Worker with augmentation + replay
- `train_v6.py` - Training script with replay buffer
- `augmentation.py` - Data augmentation utilities
- `replay_buffer.py` - Experience replay buffer

## Training Command

### Recommended (4 workers, min-len 4)

```bash
PYTHONPATH=/home/chaseungjoon/code/WildfirePrediction:$PYTHONPATH \
python3 rl_training/a3c/train_v6.py \
  --num-workers 4 \
  --max-episodes 1000 \
  --lr 7e-5 \
  --entropy-coef 0.015 \
  --min-episode-length 4 \
  --replay-capacity 100 \
  --replay-prob 0.3 \
  --replay-min-iou 0.15 \
  --wandb-project wildfire-prediction \
  --wandb-run-name a3c-v6-phase1
```

### Quick Test (50 episodes)

```bash
PYTHONPATH=/home/chaseungjoon/code/WildfirePrediction:$PYTHONPATH \
python3 rl_training/a3c/train_v6.py \
  --num-workers 4 \
  --max-episodes 50 \
  --lr 7e-5 \
  --min-episode-length 4 \
  --no-wandb
```

### Without Augmentation (Ablation Study)

```bash
PYTHONPATH=/home/chaseungjoon/code/WildfirePrediction:$PYTHONPATH \
python3 rl_training/a3c/train_v6.py \
  --num-workers 4 \
  --max-episodes 1000 \
  --no-augmentation \
  --wandb-run-name a3c-v6-no-aug
```

## Hyperparameters

### Optimal (from V3)
- `--num-workers 4` - Best balance (not 2 or 8)
- `--lr 7e-5` - Learning rate
- `--entropy-coef 0.015` - Entropy coefficient
- `--min-episode-length 4` - Episode quality filter (502 episodes)

### New in V6
- `--replay-capacity 100` - Store top 100 episodes
- `--replay-prob 0.3` - 30% sampling from replay buffer
- `--replay-min-iou 0.15` - Minimum IoU to add to buffer
- `--no-augmentation` - Disable data augmentation (for ablation)

## Expected Results

### Phase 1 Target
- **Best IoU:** 45-50%
- **Training time:** ~1 hour (1000 episodes, 4 workers)
- **Memory usage:** ~40-50% RAM (with 4 workers)

### Breakdown
- **V3 baseline:** 40.91% IoU
- **+ Data augmentation:** +5-7% → ~47% IoU
- **+ Experience replay:** +2-3% → ~49% IoU

## Monitoring

### Key Metrics to Watch
1. `train/best_iou` - Should exceed 40.91% (V3 baseline)
2. `replay/buffer_size` - Should grow to 100
3. `replay/mean_iou` - Average IoU of replayed episodes
4. `worker_X/source_replay` vs `worker_X/source_filtered` - Replay vs new episodes

### WandB Logging
If using `--wandb-project`, monitor:
- Training curves (IoU, loss, rewards)
- Replay buffer statistics
- Per-worker performance

## Evaluation

After training, evaluate on validation set:

```bash
PYTHONPATH=/home/chaseungjoon/code/WildfirePrediction:$PYTHONPATH \
python3 rl_training/a3c/evaluate.py \
  --checkpoint rl_training/a3c/checkpoints_v6/YYMMDD-HHMM/best_model.pt \
  --model-type v6 \
  --max-envs 100 \
  --output results_v6.json
```

Note: Use `--model-type v6` or `--model-type v2` (they're compatible).

## Checkpoints

Saved to: `rl_training/a3c/checkpoints_v6/YYMMDD-HHMM/`

Files:
- `best_model.pt` - Best model by IoU (saved during training)
- `final_model.pt` - Final model after all episodes

## Troubleshooting

### Memory Issues (99% RAM)
- Reduce `--num-workers` to 2
- Increase `--max-file-size-mb` to skip large files
- Reduce `--replay-capacity` to 50

### Low Performance (< 40% IoU)
- Check episode filtering: ensure 400-600 episodes with `--min-episode-length 4`
- Verify augmentation is enabled (no `--no-augmentation` flag)
- Check replay buffer is filling up (`replay/buffer_size` > 50)

### Training Too Slow
- Reduce `--max-episodes` to 500
- Reduce `--min-episode-length` to 3 (more episodes, but lower quality)

## Next Steps (Phase 2)

After V6 reaches 45-50% IoU, implement:
1. **Spatial Attention** - Focus on fire front regions
2. **Temporal LSTM** - Process fire history (last 3-5 timesteps)
3. **Multi-scale Features** - Dilated convolutions for larger receptive field

See `IMPROVEMENTS.md` for full roadmap.

## Comparison with V3

| Feature | V3 | V6 |
|---------|----|----|
| Architecture | 417K params | 417K params (same) |
| Data Augmentation | ❌ | ✅ (8x dataset) |
| Experience Replay | ❌ | ✅ (top 100 episodes) |
| Episode Filtering | min-len 4 (502 eps) | min-len 4 (502 eps) |
| Workers | 4 | 4 |
| Best IoU | 40.91% | **Target: 45-50%** |

## References

- V3 baseline: 40.91% IoU
- PROGRESS.md: Full training history
- IMPROVEMENTS.md: Complete improvement roadmap
- TODO.md: Sequential testing plan

---

**Created:** 2025-11-15
**Status:** Ready for training
**Expected improvement:** +10-20% relative gain over V3
