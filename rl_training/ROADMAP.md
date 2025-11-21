# Wildfire Prediction Model Roadmap

## Current Status
- **Best Model:** A3C V3 (8-neighbor, 417K params)
- **Best Performance:** 40.91% IoU (4 workers, min-len 4)
- **Target:** 70% IoU

- **Data:** Re-embedded and re-tiled weather data (14 channels -> 15), need to tweak models accordingly

---

## Phase 1: Quick Wins (Target: 45-50% IoU, 1-3 days)

### 1.1 Data Augmentation 
**Problem:** Only 502 high-quality episodes available

**Solution:**
- Spatial augmentation: 90°, 180°, 270° rotations, horizontal/vertical flips
- Effective dataset size: 502 → 2000+ episodes
- Fire spread physics are rotation/flip invariant

**Status:** Implemented in V6
**Result:** Modest gain to 36.36% IoU

---

### 1.2 Experience Replay Buffer
**Problem:** Catastrophic forgetting - model forgets good episodes

**Solution:**
- Store top 100 episodes (sorted by IoU)
- Replay 30% of training from buffer
- Balance new exploration with exploitation of known good experiences

**Expected Gain:** +3-5% IoU (reduces forgetting)
**Difficulty:** Easy
**Status:** Implemented in V6
**Result:** Modest gain to 36.36% IoU

---

## Phase 2: Architecture Enhancements

### 2.1 Spatial Attention Mechanism
**Problem:** Model treats all regions equally, doesn't focus on fire front

**Solution:**
```python
class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention_map = torch.sigmoid(self.conv(torch.cat([avg_pool, max_pool], dim=1)))
        return x * attention_map
```

**Benefits:**
- Focus on fire boundaries (where spread happens)
- Ignore inactive regions
- Learn spatially-varying importance

**Expected Gain:** +5-8% IoU
**Difficulty:** Medium
**Status:** Planned for V8

---

### 2.2 Temporal Context (LSTM/GRU)
**Problem:** Model only sees current timestep, ignores fire history

**Current:** `obs_t` → policy
**Proposed:** `[obs_{t-2}, obs_{t-1}, obs_t]` → LSTM → policy

**V7 Attempt (FAILED - 8.13% IoU, -80% vs V3):**

**Why V7 Failed:**
1. **3D Conv ≠ LSTM**: 3D convolutions capture local spatial-temporal patterns, but lack memory/state
2. **Fire spread needs STATE**: Velocity and acceleration require hidden state (LSTM/GRU)
3. **Reduced encoder capacity**: Cut from 128 to 64 channels, lost representation power
4. **Window too small**: 3 timesteps insufficient for meaningful temporal patterns

**Correct Implementation (V7.5):**

**Critical Changes from V7:**
1. Keep FULL V3 encoder (128 channels)
2. Use LSTM (not 3D conv) - proper temporal state
3. Increase window to 5 timesteps (not 3)
4. Add layer norm for training stability
5. ~480K params total (~350K encoder + ~130K LSTM)

**Benefits:**
- Capture fire spread velocity
- Model temporal trends (accelerating/decelerating spread)
- Better wind pattern understanding
- Maintain spatial feature quality from V3

**Expected Gain:** +7-10% IoU (if done correctly)
**Difficulty:** Medium
**Status:** Implemented in V7.5
**Result:** 0.092 IoU

---

### 2.4 Channel Attention 
**Problem:** All 128 feature channels weighted equally

**Solution:**
```python
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, H, W)
        gap = x.mean(dim=[2, 3])  # (B, C)
        weights = self.fc(gap).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * weights
```

**Benefits:**
- Dynamically weight important features (e.g., wind > humidity in dry conditions)
- Learn context-dependent feature importance

**Expected Gain:** +3-5% IoU
**Difficulty:** Easy
**Status:** Can be implemented in V8.5 
---

## Phase 3: Advanced Training Strategies

### 3.1 Curriculum Learning
**Problem:** Training on mixed difficulty episodes leads to unstable learning

**Strategy:**
1. **Stage 1 (eps 0-200):** min-len 2-3 episodes (easy, lots of data)
2. **Stage 2 (eps 200-500):** min-len 3-4 episodes (medium difficulty)
3. **Stage 3 (eps 500+):** min-len 4-5 episodes (hard, production-like)

**Benefits:**
- Stable early learning
- Gradual difficulty increase
- Better generalization

**Expected Gain:** +4-6% IoU
**Difficulty:** Medium
**Status:** Planned for Phase 3 (P1, needs to be separated from V1~V8)

---

### 3.2 Prioritized Experience Replay
**Problem:** Uniform sampling from replay buffer suboptimal

**Solution:**
- Priority = f(IoU, recency, diversity)
- Sample hard episodes more frequently (IoU < 20%)
- Balance exploration (low IoU) with exploitation (high IoU)

**Implementation:**
```python
def compute_priority(episode_iou, episode_age, episode_diversity):
    # Hard episodes (low IoU) get higher priority
    difficulty_bonus = 1.0 / (episode_iou + 0.1)

    # Recent episodes get higher priority
    recency_bonus = 1.0 / (episode_age + 1)

    # Diverse episodes (rare fire patterns) get higher priority
    return difficulty_bonus * 0.5 + recency_bonus * 0.3 + episode_diversity * 0.2
```

**Expected Gain:** +3-4% IoU
**Difficulty:** Medium
**Status:** Planned for Phase 3 (P2)

---

### 3.3 Reward Shaping
**Current Reward:** Pure IoU (sparse, binary)

**Proposed Multi-Component Reward:**
```python
def compute_reward(pred_mask, actual_mask, fire_front_mask):
    # Component 1: IoU (primary)
    iou = compute_iou(pred_mask, actual_mask)  # Weight: 0.6

    # Component 2: Fire front accuracy (bonus for predicting active spread zones)
    front_acc = (pred_mask & fire_front_mask).sum() / (fire_front_mask.sum() + 1e-8)  # Weight: 0.2

    # Component 3: Precision penalty (avoid false positives)
    precision = (pred_mask & actual_mask).sum() / (pred_mask.sum() + 1e-8)  # Weight: 0.2

    return 0.6 * iou + 0.2 * front_acc + 0.2 * precision
```

**Expected Gain:** +2-4% IoU
**Difficulty:** Medium
**Status:** Separate from model architecture, can be implemented in V1~V8, P1~

---

### 3.4 Learning Rate Schedule
**Current:** Fixed lr = 7e-5

**Options:**

**A) Cosine Annealing:**
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=1000, eta_min=1e-6
)
```

**B) Reduce on Plateau:**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=100
)
```

**C) Warmup + Decay:**
- Episodes 0-100: Linear warmup 1e-6 → 1e-4
- Episodes 100-1000: Cosine decay 1e-4 → 1e-6

**Expected Gain:** +2-3% IoU
**Difficulty:** Easy
**Status:** Planned for Phase 3

---

### 3.5 Ensemble Methods HIGH IMPACT
**Strategy:** Train 5 models with different random seeds

**Ensemble Approaches:**

**A) Voting Ensemble:**
```python
# Each model predicts binary mask, majority vote
ensemble_pred = (model1_pred + model2_pred + ... + model5_pred) >= 3
```

**B) Probability Averaging:**
```python
# Average predicted probabilities before thresholding
ensemble_prob = (model1_prob + ... + model5_prob) / 5
ensemble_pred = ensemble_prob > 0.5
```

**C) Weighted Ensemble:**
```python
# Weight by validation IoU
weights = [0.25, 0.23, 0.21, 0.18, 0.13]  # Sorted by performance
ensemble_prob = sum(w * p for w, p in zip(weights, model_probs))
```

**Expected Gain:** +5-8% IoU (reliable improvement)
**Difficulty:** Easy (just train 5x)
**Status:** To be implemented in P3

---

## Phase 4: Advanced Architectures (Target: 65-70% IoU, 3-4 weeks)

### 4.1 Graph Neural Network (GNN) Approach
**Problem:** Fire spread is fundamentally a graph problem (cell-to-cell propagation)

**Reformulation:**
- Nodes = grid cells
- Edges = neighbor connectivity (8-connected or 4-connected)
- Node features = environmental features (14 channels)
- Message passing = fire spread between neighbors

**Architecture:**
```python
class FireSpreadGNN(nn.Module):
    def __init__(self):
        self.node_encoder = MLP([14, 64, 128])
        self.gnn_layers = nn.ModuleList([
            GCNConv(128, 128) for _ in range(3)
        ])
        self.edge_predictor = MLP([128*2, 64, 1])  # Predict spread probability
```

**Benefits:**
- Naturally models cell-to-cell spread
- Explicit neighbor relationships
- Can handle irregular grids

**Expected Gain:** +5-10% IoU (research direction)
**Difficulty:** Hard (new paradigm)
**Status:** Research exploration

---

## Phase 5: Data & Problem Formulation (Ongoing)

### 5.1 Better Episode Quality Metrics
**Current:** Filter by min-episode-length only

**Proposed:** Multi-criteria filtering
- Fire spread diversity (not just straight-line spread)
- Terrain complexity (elevation variance, slope changes)
- Weather variability (wind shifts during episode)
- Fire intensity distribution

**Scoring Function:**
```python
def episode_quality_score(episode):
    diversity = measure_spread_patterns(episode)
    terrain_complexity = std(elevation) + std(slope)
    weather_var = std(wind_speed) + std(wind_direction)
    intensity_range = max(intensity) - min(intensity)

    return 0.3 * diversity + 0.3 * terrain_complexity + 0.2 * weather_var + 0.2 * intensity_range
```

**Expected Gain:** +3-5% IoU
**Difficulty:** Medium
**Status:** Can implement anytime

---

### 5.2 Synthetic Episode Generation
**Problem:** Real wildfire data is limited

**Approach A - Physics-Based Simulation:**
- Use Rothermel fire spread model
- Generate synthetic episodes with controlled parameters
- Augment real data with synthetic data

**Approach B - GAN-Based Generation:**
- Train GAN to generate realistic fire spread sequences
- Condition on terrain, weather
- Generate diverse episodes

**Expected Gain:** +5-8% IoU (if synthetic data is realistic)
**Difficulty:** Very Hard
**Status:** Research exploration

---

## Phase 6: Evaluation & Analysis (Ongoing)

### 6.1 Comprehensive Validation Evaluation
**Missing:** Systematic validation set testing

**Needed:**
- Run `evaluate.py` on best checkpoints
- Report mean/std/median IoU, F1, Precision, Recall
- Compare across model versions (V3, V5, V6, etc.)

**Metrics to Track:**
```python
metrics = {
    'iou': {'mean': X, 'std': Y, 'median': Z},
    'f1': {...},
    'precision': {...},
    'recall': {...},
    'per_episode_breakdown': [...]
}
```

**Status:** Tools available (`evaluate.py`), need to run

---

### 6.2 Failure Case Analysis
**Question:** Where does the model fail?

**Categories to Analyze:**
- **Wind-driven fires:** Does model handle strong/shifting winds?
- **Terrain boundaries:** Does model respect terrain barriers (rivers, cliffs)?
- **Rapid spread:** Does model predict explosive fire growth?
- **Slow spread:** Does model avoid false positives in slow conditions?

**Approach:**
1. Sort validation episodes by IoU (low to high)
2. Manually inspect bottom 10% (worst predictions)
3. Categorize failure modes
4. Design targeted improvements

**Status:** Not yet done

---

### 6.3 Visualization & Interpretability
**Needed:**
- Visualize predictions vs ground truth (animated GIFs)
- Show attention maps (if using attention)
- Feature importance analysis
- Error heatmaps (where does model fail spatially?)

**Tools:**
```python
def visualize_prediction(env, model, timestep):
    # Show: terrain, fire_t, fire_{t+1}_actual, fire_{t+1}_predicted
    # Highlight: TP (green), FP (red), FN (yellow), TN (gray)
```

**Status:** Not yet implemented

---

## Implementation Roadmap

### Completed
- V1-V2: Initial formulation (failed)
- V3: Per-cell 8-neighbor formulation (40.91% IoU)
- V3.5: Per-pixel LSTM context (12.76% IoU) 
- V4: Worker count optimization (4 workers optimal)(40.1% IoU)
- V5: 4-neighbor multi-task (failed - 12% F1)
- V6: Data augmentation + replay (36.36% IoU)
- V7: 3D Conv temporal (8.13% IoU)
- V7.5 : Temporal context with LSTM (9.2% IoU)
- Episode quality filtering (min-len 2→3→4)

### Next Up 
- **V8:** Spatial(+Channel) attention mechanism

### Learning + Training Methods
- Multi-scale features
- Curriculum learning
- Ensemble methods
- Advanced architectures (GNN, Transformer)

---

## What we've learned so far

1. **Episode Quality > Quantity:** 502 good episodes beats 5036 mixed episodes
2. **Don't Sacrifice Capacity:** Keep full encoder (128 channels), performance > efficiency
3. **Validate Frequently:** Don't overtrain on training set

---

## References & Inspiration

- **MCTS-A3C:** 87% baseline (Saskatchewan simulation dataset)
- **U-Net Baseline:** 19.85% F1 (~11% IoU) - supervised learning ceiling
- **V3 Breakthrough:** 40.91% IoU - correct problem formulation matters
- **Worker Count Discovery:** 4 workers optimal (not 2 or 8)

---

**Last Updated:** 2025-11-21 (Post re-embed and re-tile)
**Target:** 70% IoU for production deployment
