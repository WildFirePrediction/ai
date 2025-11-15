# A3C Training TODO - Path to 70% IoU

## Current Status
- **Breakthrough:** 4 workers (not 8) is the sweet spot - prevents memory bottleneck
- **V2 Model Completed:** 40.91% IoU at episode 731 (27% improvement over 8 workers)
- **Best Config:** min-episode-length 4 (502 episodes), lr 7e-5, entropy 0.015, 4 workers
- **Checkpoint:** `checkpoints_v3/`

---

## Sequential Testing Plan

> **Step 1: V3 Model + 4 Workers + Min-Len 4 (✓ COMPLETED)**
>
> **Why:** Establish baseline with optimal worker count
> 
> **Result:** 40.91% IoU at episode 731/1000
> 
> **Status:** Complete - Model plateaued after peak, stable 27-30% IoU for eps 800-1000

---

> **Step 2: Medium Model + 4 Workers + Min-Len 4 (✓ COMPLETED)**
>
> **Expected:** 45-55% IoU
>
> **Why:** 935K params (2.2x V2) should capture more complex fire spread patterns. Memory-safe with 4 workers.
>
> **Result:** Under 10% IoU after 600 episodes, likely due to overfitting

---

> **Step 2.5: V3 Model + 2 Workers + Min-Len 4 (✓ COMPLETED)**
>
> **Why:** Test if fewer workers help with improvement with overfitting issue
>
> **Result:** IoU under 5% after 500 episodes - fewer workers worsened overfitting

---

> **Step 3: V5 (4-Neighbor Multi-Task) + 4 Workers + Min-Len 4**
>
> **Why:**
> - Simpler action space: 4 neighbors (N,E,S,W) vs 8 - diagonal spread likely noise
> - Multi-task learning: burn + intensity + temperature prediction - richer gradients
> - Better metrics: F1, precision, recall (not just IoU)
> - Inspired by MCTS-A3C (87% baseline)
> 
> **Result** : 10% F1 Score


> **Step 3.5: V5 (4-Neighbor Multi-Task) + 8 Workers + Min-Len 4**
>
> **Why:**
> - Simpler action space: 4 neighbors (N,E,S,W) vs 8 - diagonal spread likely noise
> - Multi-task learning: burn + intensity + temperature prediction - richer gradients
> - Better metrics: F1, precision, recall (not just IoU)
> - Inspired by MCTS-A3C (87% baseline)
>
> **Result** : 12% F1 Score

---

> **Step 4: V7 (Temporal 3D Conv) + 4 Workers + Min-Len 4 (✓ COMPLETED - FAILED)**
>
> **Why:**
> - Add temporal context (last 3 timesteps) to capture fire velocity/acceleration
> - Target: 47-50% IoU improvement over V3's 40.91%
>
> **Architecture:**
> - Lighter CNN encoder: 14 → 32 → 64 channels (vs V3's 128)
> - 3D Conv for temporal modeling (changed from planned LSTM)
> - Window size: 3 timesteps
> - ~450K params vs V3's 417K
>
> **Result: CATASTROPHIC FAILURE**
> - Best IoU: **8.13%** at episode 560
> - **-80% degradation** vs V3 baseline (40.91% → 8.13%)
>
> **Why It Failed:**
> 1. Reduced encoder capacity (64 vs 128 channels) - lost representation power
> 2. 3D Conv wrong tool for temporal memory (needs LSTM state for velocity tracking)
> 3. Window size too small (3 timesteps insufficient)
> 4. Optimized for "lighter" model instead of better temporal modeling
>
> **Key Lesson:** Temporal modeling requires recurrent architecture (LSTM/GRU) with hidden state, not 3D convolutions. Don't sacrifice encoder capacity for efficiency.

---

## Key Insights

**What Worked:**
1. **4 workers** - prevents memory bottleneck (99% RAM → 40-50%)
2. **min-episode-length 4** - 502 high-quality episodes (sweet spot)
3. **Episode quality > quantity** - filtering was critical
4. **Model capacity matters** - but not at expense of memory

**What NOT to Try:**
- ~~8 workers~~ - causes memory explosion with any model, overfitting worsens
- ~~min-episode-length 5-6~~ - too sparse (90 episodes), not enough data for 4 workers
- ~~Deep 5-layer model~~ - 4M params too heavy for CPU training
- ~~3D Conv for temporal modeling~~ - V7 failed catastrophically (8.13% IoU vs 40.91%)
- ~~Reducing encoder capacity~~ - Don't sacrifice 128-channel layer for "efficiency"

---

## Success Criteria

- **V3 Target:** 40-50% IoU (✓ Achieved: 40.91%)
- ~~**Medium Target:** 45-55% IoU~~ 10% IoU (overfitting issue)
- **V5 Target:** 50-60% IoU (Failed: 12% F1)
- **V7 Target:** 47-50% IoU (Failed catastrophically: 8.13% IoU, -80% vs V3)
- **Production Goal:** 70% IoU (may need ensemble/attention after this)

---

## Model Specs

| Model  | Params | Architecture | Key Feature | Result |
|--------|--------|--------------|-------------|--------|
| V3     | 417K | 3-layer (32→64→128) | Baseline, proven stable | ✅ 40.91% IoU |
| Medium | 935K | 3-layer (48→96→192) + GroupNorm | 2.2x capacity, CPU-friendly | ❌ <10% IoU (overfitting) |
| V5     | 497K | 3-layer + multi-task | 4-neighbor + burn/intensity/temp | ❌ 12% F1 |
| V7     | ~450K | 2-layer (32→64) + 3D Conv temporal | Window=3, lighter encoder | ❌ 8.13% IoU (-80% vs V3) |

---

## Inference & Evaluation

> **Single Environment Inference**
>
> **Command:**
> ```bash
> PYTHONPATH=/home/chaseungjoon/code/WildfirePrediction:$PYTHONPATH \
> python3 rl_training/a3c/inference.py \
>   --checkpoint rl_training/a3c/checkpoints_v3/YYMMDD-HHMM/best_model.pt \
>   --model-type v2 \
>   --env-path tilling_data/environments/some_env.pkl \
>   --start-t 0 \
>   --max-steps 20
> ```
>
> **What it does:** Runs trained model on single environment, prints step-by-step metrics (IoU, F1, Precision, Recall).

---

> **Validation Set Evaluation**
>
> **Command:**
> ```bash
> PYTHONPATH=/home/chaseungjoon/code/WildfirePrediction:$PYTHONPATH \
> python3 rl_training/a3c/evaluate.py \
>   --checkpoint rl_training/a3c/checkpoints_v3/YYMMDD-HHMM/best_model.pt \
>   --model-type v2 \
>   --max-envs 100 \
>   --max-steps 20 \
>   --output results_v2.json
> ```
>
> **What it does:** Tests model on validation set, computes mean/std/min/max/median metrics, saves to JSON.

---

**Model Type Options:**
- `v2` - Original model (417K params)
- `medium` - Medium capacity (935K params)
- `v5` - 4-neighbor multi-task (497K params)

---

**Last Updated:** 2025-11-16 (V7 Postmortem)
