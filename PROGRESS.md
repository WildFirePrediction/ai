# Wildfire Prediction Training Progress

## Best Scores Summary

### Supervised Learning (F1 Score)
> **U-Net** V2 : 0.1985 (@epoch 3)

### Reinforcement Learning (A3C) (IoU Score)
> **V3** --min-episode-length **2**, --num-workers **8** : **0.094**
>
> **V3** --min-episode-length **3**, --num-workers **8** : **0.1786** (@episode 305)
>
> **V3** --min-episode-length **4**, --num-workers **8** : **0.4000** (@episode 1918)
> 
> **V3** --min-episode-length **5**, --num-workers **8** : **0.1165** (@episode 428)
>
> **V3** --min-episode-length **4**, --num-workers **4** : **0.4091** (@episode 731)
>
> **V3** --min-episode-length **4**, --num-workers **2** : **0.0569** (@episode 76)
>
> **V3.5** --mel **4**, --num-workers **4** - CRASHED (Memory overflow)
>
> **V3.5-Fixed** --mel **4**, --num-workers **1** - **0.1276** (@episode 61)
>
> **V3.5-Fixed** --mel **4**, --num-workers **2** - **0.0873** (@episode 820)

> **V5** --min-episode-length **4**, --num-workers **4** : **0.1000** (@episode 831)
>
> **V5** --min-episode-length **4**, --num-workers **8** : **0.1242** (@episode 1790)

> **V6** --min-episode-length **4**, --num-workers **4** : **0.3636** (@episode 761)

> **V7** --min-episode-length **4**, --num-workers **4** : **0.0813** (@episode 560)
> 
> **V7.5** --min-episode-length **4**, --num-workers **4** : **0.09262** (@episode 982)

---

# RL

---

## A3C V1-V2 : CRITICAL ERROR

**Problem Formulation (WRONG):**
```
State: (14, H, W) environmental features
Action: (H, W) binary mask - predict ALL 30,000 cells independently
Reward: Sparse - only at episode end
```

**Wrong Formulation**

1. **Perdict all 30,000 cells independently
2. **Only receive reward at episode end** -> Credit assignment impossible

**Result**
- Avg reward: -4 ~ -9
- IoU: 0.0001% 
- Loss: -180,000 ~ -860,000 (unstable)

---

## A3C V7: Temporal Context with 3D Convolutions - FAILED

**Goal:** Add temporal context to V3 (40.91% IoU) → Target 47-50% IoU

**Problem Formulation:**
```
State: (B, 3, 14, H, W) - last 3 timesteps
Action: Same as V3 - 8-neighbor prediction per burning cell
Reward: Dense IoU at every timestep
Temporal Modeling: 3D Convolution (changed from planned LSTM)
```

**Architecture Changes from V3:**

1. **Lighter CNN Encoder** (MISTAKE #1)
   - V3: 14 → 32 → 64 → 128 channels
   - V7: 14 → 32 → 64 channels (NO 128!)
   - Rationale: "3D conv will compensate" - it didn't

2. **Temporal 3D Convolution** (MISTAKE #2)
   - Original plan: 2-layer LSTM (128 hidden dim)
   - Actual: 3D Conv layers (64 → 96 → 128)
   - Changed because: "MUCH lighter than LSTM!"
   - Problem: 3D convs don't capture temporal dynamics like LSTM

3. **Temporal Window**
   - Window size: 3 timesteps
   - Padding: Repeat first observation for t < 3
   - Memory: 3x feature maps (42 vs 14 in V3)

4. **Same Policy/Value Heads as V3**
   - Policy: Per-cell 8-neighbor prediction
   - Value: Global average pooling + FC

**Training Configuration:**
```
Workers: 4 (same as V3)
Learning rate: 7e-5 (same as V3)
Entropy coef: 0.015 (same as V3)
Min episode length: 4 (502 filtered episodes)
Max episodes: 5000
Window size: 3 timesteps
```

**Results: CATASTROPHIC FAILURE**

| Metric | V7 Result | V3 Baseline | Performance |
|--------|-----------|-------------|-------------|
| **Best IoU** | **8.13%** | **40.91%** | **-80% degradation** |
| Episode | 560 | 731 | - |
| Model params | ~450K | 417K | +8% |
| Training time | ~1 hour | ~1 hour | Same |

**Why It Failed So Badly:**

1. **Reduced Feature Capacity**
   - Cut 128-channel layer to save computation
   - Lost critical representation power
   - 3D conv couldn't compensate for weaker encoder

2. **3D Conv vs LSTM Trade-off**
   - 3D convs: Capture spatial-temporal patterns in receptive field
   - LSTMs: Capture long-term temporal dependencies and state
   - Fire spread needs MEMORY (velocity, acceleration) not just local patterns
   - 3D conv was wrong tool for this task

3. **Optimization Issues**
   - 3x memory usage (temporal window)
   - More parameters but worse performance
   - Gradients likely had trouble flowing through 3D conv layers

4. **Window Size Too Small**
   - 3 timesteps insufficient to learn fire velocity/acceleration
   - Need 5-7 timesteps to see meaningful temporal patterns

5. **Wrong Optimization Target**
   - Optimized for "lighter" model
   - Should have optimized for "better temporal modeling"
   - Penny-wise, pound-foolish

**Critical Insights:**

- **Temporal modeling needs STATE**: LSTM/GRU maintain hidden state, 3D conv doesn't
- **Don't sacrifice encoder capacity**: The 128-channel layer in V3 was critical
- **Memory is a feature, not a bug**: LSTM's state memory is exactly what we need
- **"Lighter" doesn't mean "better"**: Performance > efficiency at this stage

**What Should Have Been Done (V7.5 Plan):**

1. **Keep full V3 encoder** (32 → 64 → 128)
2. **Use 2-layer LSTM** as originally planned (not 3D conv)
3. **Increase window size** to 5 timesteps (not 3)
4. **Add layer norm** to stabilize LSTM training
5. **Hybrid approach**: CNN encoder → LSTM temporal → V3 heads

**Expected params for correct V7:**
- V3 encoder: ~350K params
- 2-layer LSTM (128 hidden): ~130K params
- Total: ~480K params (worth it for temporal modeling)

---

## A3C V3: Correct Formulation

**Insight :** Predict 8-neighbor spread for each currently burning cell

> **Problem Formulation**
```
State: (14, H, W) environmental features + current fire mask
Action: For each burning cell at (i,j), predict 8-neighbor spread
        - 8-dimensional Bernoulli vector per burning cell
        - Neighbors: N, NE, E, SE, S, SW, W, NW
        - Total actions per step: K burning cells × 8 neighbors (~10-100)
Reward: DENSE - IoU computed at EVERY timestep
```

> **Architecture Changes**

1. **Shared CNN Encoder**
   - Input: (B, 14, H, W)
   - Output: (B, 128, H, W) feature map

2. **Per-Cell Policy Head**
   - Extract 3×3 local features around each burning cell
   - Flatten to 128×9 = 1152 dimensions
   - FC layers: 1152 → 256 → 64 → 8
   - Output: 8-dim logits for neighbor predictions

3. **Global Value Head**
   - Adaptive average pooling over full feature map
   - FC layers: 128 → 64 → 1
   - Output: Scalar state value

> Status and Results

## Training Configuration

**A3C V3 Settings:**
```
Model: A3C_PerCellModel (416,873 parameters)
Workers: 8(or 4) parallel CPU workers
Episodes: 1000 ~ 3000 
Learning Rate: 1e-4
Gamma (discount): 0.99
Value Loss Coef: 0.5
Entropy Coef: 0.01
Max Grad Norm: 0.5
```

## Performance Metrics

| Approach                          | Best Performance        | Training Time | Issues                                     |
|-----------------------------------|-------------------------|---------------|--------------------------------------------|
| U-Net V2 (pos=150)                | F1: 19.85%, IoU ~11-12% | 3 epochs, ~1 hour | Early peaking, overfitting                 |
| U-Net V3 (Combined)               | F1: 15.95%              | 5 epochs, ~1.5 hours | Over-prediction                            |
| U-Net V3 (Dice)                   | F1: 6.68%               | 3 epochs, ~1 hour | Severe over-prediction                     |
| A3C V3 (min-len 2)                | IoU: 9.4%               | 1000 episodes, ~1 hour | Unstable, catastrophic forgetting          |
| A3C V3 (min-len 3)                | IoU: 17.86%             | 1000 episodes, ~1 hour | Stable, some forgetting after peak         |
| **A3C V3 (min-len 4, 8 workers)**            | **IoU: 32.14%**         | **1000 episodes, ~1 hour** | **Very stable, no forgetting**             |
| **A3C V3 (min-len 4, 4 workers)** | **IoU: 40.91%**         | **1000 episodes, ~1 hour** | **Fixes overfitting issue with 8 workers** |
| A3C V7 (3D Conv, 4 workers)       | IoU: 8.13%              | 560 episodes, ~1 hour | **FAILED: -80% vs V3, wrong architecture** |

A3C V3 with episode quality filtering achieves 40.91% IoU, nearly 4x better than supervised learning's best (11-12% IoU equivalent). V7's failure demonstrates that temporal modeling requires proper recurrent architecture (LSTM/GRU), not 3D convolutions.

**Key Observations:**

1. **Stable Training:** Loss remains in reasonable range without explosions
2. **Clear Learning:** IoU progression from 0% to 14% shows model is learning
3. **High Variance:** Individual episodes range from 0% to 14% (normal for RL)
4. **Parallel Execution:** All 8 workers contributing (confirmed by rapid episode completion)
5. **Dense Reward Signal:** Receiving feedback at every timestep, not just episode end

---

## A3C V4 : V3 with increased parameters (4M)(CRASHED)

## A3C V3-medium : V3 with medium parameter count (917k)

### Result

-  **IoU :** : 0.7

## A3C V5 : 4 neighbor instead of 8

### Result

- **IoU :** : 0.12

## A3C V6 : Data augmentation 

### Result

- **IoU :** : 0.36

## A3C V7 : 3D Conv architecture

### Result 

- **IoU :** : 0.09

## A3C V3.5 : LSTM 

### Result

- **IoU :** : -

---

# Supervised Learning (U-Net)

**Problem:** Extreme class imbalance (31,077:1)
- 79.1% samples have zero burns
- Model learns "trivial solution" (predict all zeros)

**Attempts:**
- BCE pos_weight=150: **19.85% F1** (best, epoch 3)
- Combined Loss: 15.95% F1
- Dice Loss: 6.68% F1
- All peaked early (epoch 3-5), then overfitted

**Conclusion:** 20% F1 (~11% IoU) is ceiling for supervised learning on this task.

---

## File Locations

**Supervised:**
- Models: `rl_training/supervised/unet_model*.py`
- Training: `rl_training/supervised/train_unet*.py`
- Best: `checkpoints_unet_v2/best_model.pt` (F1=19.85%)

**Reinforcement Learning:**
- Model: `rl_training/a3c/model_v2.py` (V3, 417K params)
- Worker: `rl_training/a3c/worker_v3.py`
- Training: `rl_training/a3c/train_v3.py`
- Best: `checkpoints_v3/251112-2206/best_model.pt` (IoU=40.91%)

**Data:**
- Environments: `tilling_data/environments/*.pkl` (3026 train, 648 val)
- Observation: 14 channels (static features, fire state, weather)
- Per-cell 8-neighbor prediction

---

## Training Results

### Worker Count Optimization (2025-11-12)

**Discovery:** Worker count is the critical bottleneck, not model size.

**Results:**

| Configuration | Workers | Params | Peak IoU | Episode | RAM |
|--------------|---------|--------|----------|---------|-----|
| V3 + min-len 4 | 8 | 417K | 32.14% | 960 | 99% |
| V3 + min-len 4 | **4** | 417K | **40.91%** | 731 | 40% |
| V3 + min-len 4 | 2 | 417K | 5.69% | 76 | Low |
| Medium + min-len 4 | 4 | 935K | <10% | 600 | 40% |

**Key Findings:**
- **4 workers is optimal:** Best balance of memory and parallelism
- **2 workers:** Too little parallelism, poor exploration
- **8 workers:** Memory bottleneck, swap thrashing
- **Larger model (935K):** Overfitting, didn't help
- **Best checkpoint:** `checkpoints_v3/mel4-workers4(0.4091)/best_model.pt`

---

### Episode Quality Filtering

**Discovery:** Episode quality > hyperparameter tuning.

| min-len | Episodes | Peak IoU | Improvement |
|---------|----------|----------|-------------|
| 2 | 5036 | 9.4% | Baseline |
| 3 | 1884 | 17.86% | 1.9x |
| 4 (8 workers) | 502 | 32.14% | 1.8x |
| 4 (4 workers) | 502 | **40.91%** | 1.3x |
| 5 (8 workers) | ~90 | 11.65% | Failed (too sparse) |

**Key Findings:**
- Quality > quantity: 502 episodes beats 5036
- min-len 4 is sweet spot (502 episodes)
- min-len 5+ too sparse, not enough data
- Best config: min-len 4, 4 workers, lr 7e-5, entropy 0.015

### Next Steps

**Completed:**
- ✓ V3 (417K) + 4 workers + min-len 4: **40.91% IoU**
- ✓ Medium (935K) + 4 workers: Failed (overfitting)
- ✓ V3 + 2 workers: Failed (5.69% IoU)
- ✓ min-len 5: Failed (too sparse, 11.65% IoU)

**Next:**
1. Test V5 (4-neighbor multi-task, 497K params) + 4 workers
2. Try attention mechanisms or multi-scale features
3. Evaluate best model on validation set
4. Target: 50-60% IoU (production goal: 70%)

---

## Lessons Learned

### What Worked
1. **Episode quality filtering (KEY):** min-len 4 = sweet spot (502 episodes)
2. **Worker count:** 4 workers optimal (not 2 or 8)
3. **Dense rewards:** IoU at every timestep
4. **Per-cell 8-neighbor prediction:** Matches fire physics
5. **Checkpoint saving:** Preserves best model

### What Didn't Work
1. **8 workers:** Memory bottleneck (99% RAM)
2. **2 workers:** Poor exploration
3. **Larger model (935K params):** Overfitting
4. **min-len 5+:** Too sparse, not enough data
5. **Full grid prediction:** 30K action space failed
6. **V7 (3D Conv temporal):** -80% performance vs V3
   - Reduced encoder capacity (64 vs 128 channels)
   - 3D conv instead of LSTM (wrong tool for temporal memory)
   - Window size too small (3 timesteps insufficient)

---

## Conclusion

**Progress:**
- Supervised (U-Net): 19.85% F1
- A3C V3 (417K params, 4 workers, min-len 4): **40.91% IoU** ← Current best
- A3C V7 (3D Conv temporal): 8.13% IoU ← FAILED

**Key Insights:**
1. Episode quality > quantity (502 episodes beats 5036)
2. Worker count critical (4 optimal, not 2 or 8)
3. Model capacity not the bottleneck (935K params overfits)
4. Per-cell 8-neighbor prediction matches fire physics
5. Dense rewards essential for RL
6. **Temporal modeling needs recurrent state (LSTM/GRU), not 3D convolutions**
7. **Don't sacrifice encoder capacity for "efficiency" - performance first**

**Path to 70% IoU:**
- Current: 40.91%
- Next: V5 (4-neighbor multi-task) targeting 50-60%
- Architecture improvements: +5-10%
- Total: 55-70% achievable

---

**Repository:** `/home/chaseungjoon/code/WildfirePrediction`

**Last Updated:** 2025-11-16 (V7 Postmortem)
