# Wildfire Prediction Model Improvement Options

## Current Status
- **Best Model:** A3C V3 (8-neighbor, 417K params)
- **Best Performance:** 40.91% IoU (4 workers, min-len 4)
- **Target:** 70% IoU for production
- **Gap to Close:** ~30% IoU improvement needed

---

## Phase 1: Quick Wins (Target: 45-50% IoU, 1-3 days)

### 1.1 Data Augmentation ⭐ HIGH IMPACT
**Problem:** Only 502 high-quality episodes available

**Solution:**
- Spatial augmentation: 90°, 180°, 270° rotations, horizontal/vertical flips
- Effective dataset size: 502 → 2000+ episodes
- Fire spread physics are rotation/flip invariant

**Implementation:**
```python
def augment_episode(obs, fire_mask, burn_mask):
    # Random rotation (0, 90, 180, 270 degrees)
    # Random flip (horizontal, vertical)
    return augmented_obs, augmented_fire_mask, augmented_burn_mask
```

**Expected Gain:** +5-7% IoU
**Difficulty:** Easy
**Status:** ✅ Implemented in V6

---

### 1.2 Experience Replay Buffer ⭐ HIGH IMPACT
**Problem:** Catastrophic forgetting - model forgets good episodes

**Solution:**
- Store top 100 episodes (sorted by IoU)
- Replay 30% of training from buffer
- Balance new exploration with exploitation of known good experiences

**Implementation:**
```python
class ReplayBuffer:
    def __init__(self, capacity=100):
        self.buffer = []  # Store (env_path, start_t, IoU)

    def add_if_good(self, episode_info):
        # Keep top K episodes by IoU

    def sample(self, batch_size):
        # Sample episodes for replay
```

**Expected Gain:** +3-5% IoU (reduces forgetting)
**Difficulty:** Easy
**Status:** ✅ Implemented in V6

---

### 1.3 Improved Episode Filtering
**Problem:** min-len 4 = 502 episodes, min-len 5 = 90 episodes (too sparse)

**Options:**
- **A) Weighted Sampling:** Oversample longer episodes (min-len 4-5) 2x
- **B) Soft Filtering:** Use min-len 3 but weight by episode quality
- **C) Dynamic Threshold:** Start with min-len 3, gradually increase to 4

**Expected Gain:** +1-2% IoU
**Difficulty:** Easy
**Status:** Not yet implemented

---

### 1.4 Gradient Clipping Optimization
**Current:** Max grad norm = 0.5

**Options:**
- Try 1.0 (less aggressive clipping)
- Try adaptive clipping based on loss magnitude
- Monitor gradient norms during training

**Expected Gain:** +1-2% IoU (better gradient flow)
**Difficulty:** Trivial
**Status:** Can tune in V6

---

## Phase 2: Architecture Enhancements (Target: 50-60% IoU, 1-2 weeks)

### 2.1 Spatial Attention Mechanism ⭐ HIGH IMPACT
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
**Status:** Planned for V7

---

### 2.2 Temporal Context (LSTM/GRU) ⭐ CRITICAL MISSING
**Problem:** Model only sees current timestep, ignores fire history

**Current:** `obs_t` → policy
**Proposed:** `[obs_{t-2}, obs_{t-1}, obs_t]` → LSTM → policy

**Implementation:**
```python
class TemporalEncoder(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=feature_dim,
            num_layers=2,
            batch_first=True
        )

    def forward(self, feature_sequence):
        # feature_sequence: (B, T, C, H, W)
        # Process temporal dimension
        return temporal_features
```

**Benefits:**
- Capture fire spread velocity
- Model temporal trends (accelerating/decelerating spread)
- Better wind pattern understanding

**Expected Gain:** +7-10% IoU
**Difficulty:** Medium
**Status:** Planned for V7

---

### 2.3 Multi-Scale Feature Extraction
**Problem:** Limited receptive field (7x7 in V3), can't see distant terrain/wind

**Solution A - Dilated Convolutions:**
```python
# Add dilated convs to capture multi-scale context
nn.Conv2d(64, 128, kernel_size=3, dilation=1)  # 3x3
nn.Conv2d(128, 128, kernel_size=3, dilation=2)  # 5x5 effective
nn.Conv2d(128, 128, kernel_size=3, dilation=4)  # 9x9 effective
```

**Solution B - Feature Pyramid Network:**
- Process features at multiple scales (1x, 1/2x, 1/4x)
- Fuse multi-scale information
- Capture both local spread and global wind patterns

**Expected Gain:** +4-6% IoU
**Difficulty:** Medium
**Status:** Planned for V8

---

### 2.4 Channel Attention (Squeeze-and-Excitation)
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
**Status:** Can add to V7

---

## Phase 3: Advanced Training Strategies (Target: 60-65% IoU, 2-3 weeks)

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
**Status:** Planned for Phase 3

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
**Status:** Planned for Phase 3

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
**Status:** Planned for Phase 3

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

### 3.5 Ensemble Methods ⭐ HIGH IMPACT
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
**Status:** Planned for Phase 3

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

### 4.2 Transformer Architecture
**Problem:** Convolutions have limited receptive field, can't model long-range dependencies

**Vision Transformer for Wildfire:**
```python
class FireSpreadTransformer(nn.Module):
    def __init__(self):
        self.patch_embed = PatchEmbedding(patch_size=4, in_channels=14, embed_dim=256)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8),
            num_layers=6
        )
        self.spread_head = nn.Linear(256, 1)
```

**Benefits:**
- Global attention (see entire fire + terrain)
- Capture long-range wind effects
- State-of-the-art in vision tasks

**Challenges:**
- Computationally expensive (O(n²) for n patches)
- Requires more data
- Harder to train

**Expected Gain:** +5-8% IoU (uncertain)
**Difficulty:** Hard
**Status:** Research exploration

---

### 4.3 Hybrid CNN-Transformer
**Best of Both Worlds:**
- CNN encoder for local features (fire spread patterns)
- Transformer for global context (wind, distant terrain)

**Architecture:**
```python
class HybridModel(nn.Module):
    def __init__(self):
        self.cnn_encoder = CNNEncoder()  # Local features
        self.transformer = TransformerEncoder()  # Global context
        self.fusion = CrossAttention()  # Combine local + global
        self.policy_head = PolicyHead()
```

**Expected Gain:** +6-10% IoU
**Difficulty:** Hard
**Status:** Research exploration

---

### 4.4 Model-Based RL (World Model)
**Problem:** Model-free A3C doesn't plan ahead

**Solution:**
1. Learn world model: `M(s_t, a_t) → s_{t+1}`
2. Use model for planning (look ahead 3-5 steps)
3. Combine model-based planning with model-free policy

**Benefits:**
- Better long-term decisions
- Sample efficiency
- Can simulate "what-if" scenarios

**Challenges:**
- World model must be accurate
- Complex training (two models)
- Computationally expensive

**Expected Gain:** +8-12% IoU (high variance)
**Difficulty:** Very Hard
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

### 5.3 Fire Front Prediction (Alternative Formulation)
**Current:** Predict 8-neighbors for ALL burning cells

**Proposed:** Predict only for fire front (active spread zone)

**Definition of Fire Front:**
- Cells that are burning at time t
- AND have at least one non-burning neighbor
- These are the cells where spread will happen

**Benefits:**
- Smaller action space (10-30 cells instead of 50-100)
- Focus on active spread region
- Ignore stable interior fire

**Expected Gain:** +3-6% IoU (clearer credit assignment)
**Difficulty:** Medium
**Status:** Planned for testing

---

### 5.4 Hierarchical Action Space
**Problem:** Predicting 8 independent Bernoulli variables is hard

**Proposed:** Two-level hierarchy

**Level 1 - Sector Prediction:**
- Predict which of 8 sectors will have spread (binary, 8 dimensions)
- Sectors: N, NE, E, SE, S, SW, W, NW

**Level 2 - Intensity Prediction:**
- For sectors predicted to have spread, predict burn probability (continuous)

**Benefits:**
- Easier exploration (decomposed problem)
- Can learn sector patterns first, then fine-tune intensity

**Expected Gain:** +2-4% IoU
**Difficulty:** Medium
**Status:** Research idea

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

**Status:** ✅ Tools available (`evaluate.py`), need to run

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

### 6.4 Ablation Studies
**Questions:**
- How much does each feature contribute? (weather vs terrain vs fire history)
- Which architectural components matter? (depth, width, attention)
- What's the impact of each augmentation type?

**Systematic Testing:**
```
Baseline: V3 without augmentation
+ Rotation augmentation: +X% IoU
+ Flip augmentation: +Y% IoU
+ Experience replay: +Z% IoU
+ Rotation + Flip + Replay: +W% IoU (check for synergy)
```

**Status:** Not yet done

---

## Implementation Roadmap

### ✅ Completed
- V1-V2: Initial formulation (failed)
- V3: Per-cell 8-neighbor formulation (40.91% IoU)
- V4: Worker count optimization (4 workers optimal)
- V5: 4-neighbor multi-task (failed - 12% F1)
- Episode quality filtering (min-len 2→3→4)

### 🚧 In Progress (V6)
- Data augmentation (rotations, flips)
- Experience replay buffer
- Improved training stability

### 📋 Next Up (V7)
- Spatial attention mechanism
- Temporal context (LSTM)
- Channel attention

### 🔮 Future (V8+)
- Multi-scale features
- Curriculum learning
- Ensemble methods
- Advanced architectures (GNN, Transformer)

---

## Expected Performance Trajectory

| Phase | Models | Key Features | Target IoU | Timeline |
|-------|--------|--------------|------------|----------|
| ✅ Completed | V3 | 8-neighbor, episode filtering | 40.91% | Done |
| Phase 1 | V6 | V3 + augmentation + replay | 45-50% | 1-3 days |
| Phase 2 | V7 | V6 + attention + temporal | 52-58% | 1-2 weeks |
| Phase 3 | V7+ | V7 + curriculum + ensemble | 60-65% | 2-3 weeks |
| Phase 4 | V8+ | Advanced architectures | 65-70% | 3-4 weeks |

---

## Risk Assessment

### Low Risk, High Reward (Do First) ⭐
1. Data augmentation
2. Experience replay
3. Ensemble methods
4. Learning rate tuning

### Medium Risk, High Reward (Do Next)
1. Spatial attention
2. Temporal LSTM
3. Multi-scale features
4. Curriculum learning

### High Risk, High Reward (Research)
1. GNN architecture
2. Transformer architecture
3. Model-based RL
4. Synthetic data generation

### Low Reward (Skip for Now)
1. Micro-optimizations (optimizer choice, etc.)
2. Over-complicated reward shaping
3. Excessive hyperparameter tuning

---

## Key Principles

1. **Episode Quality > Quantity:** 502 good episodes beats 5036 mixed episodes
2. **Temporal Context is Critical:** Fire spread is inherently sequential
3. **Attention Helps:** Fire front is sparse, attention can focus computation
4. **Augmentation is Free Performance:** Rotation/flip invariance is guaranteed
5. **Ensemble When Unsure:** 5 models always beats 1 model
6. **Validate Frequently:** Don't overtrain on training set
7. **Understand Failures:** Analyze where model fails, design targeted fixes

---

## References & Inspiration

- **MCTS-A3C:** 87% baseline (Saskatchewan dataset)
- **U-Net Baseline:** 19.85% F1 (~11% IoU) - supervised learning ceiling
- **V3 Breakthrough:** 40.91% IoU - correct problem formulation matters
- **Worker Count Discovery:** 4 workers optimal (not 2 or 8)

---

**Last Updated:** 2025-11-15
**Current Focus:** Phase 1 - V6 Implementation
**Target:** 70% IoU for production deployment
