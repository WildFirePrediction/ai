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
> **Command:**
> ```bash
> PYTHONPATH=/home/chaseungjoon/code/WildfirePrediction:$PYTHONPATH \
> python3 rl_training/a3c/train_v5.py \
>   --num-workers 4 \
>   --max-episodes 1000 \
>   --max-file-size-mb 50 \
>   --min-episode-length 4 \
>   --log-interval 20 \
>   --lr 7e-5 \
>   --entropy-coef 0.015 \
>   --wandb-project wildfire-prediction \
>   --wandb-run-name a3c-v5-mel4-workers4
> ```
>
> **Why:**
> - Simpler action space: 4 neighbors (N,E,S,W) vs 8 - diagonal spread likely noise
> - Multi-task learning: burn + intensity + temperature prediction - richer gradients
> - Better metrics: F1, precision, recall (not just IoU)
> - Inspired by MCTS-A3C (87% baseline)

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

---

## Success Criteria

- **V3 Target:** 40-50% IoU (Achieved)
- ~~**Medium Target:** 45-55% IoU~~ 10% IoU (overfitting issue)
- **V5 Target:** 50-60% IoU
- **Production Goal:** 70% IoU (may need ensemble/attention after this)

---

## Model Specs

| Model  | Params | Architecture | Key Feature |
|--------|--------|--------------|-------------|
| V3     | 417K | 3-layer (32→64→128) | Baseline, proven stable |
| Medium | 935K | 3-layer (48→96→192) + GroupNorm | 2.2x capacity, CPU-friendly |
| V5     | 497K | 3-layer + multi-task | 4-neighbor + burn/intensity/temp |

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

**Last Updated:** 2025-11-12
