# A3C V7 - Temporal LSTM Architecture

## Goal
Add temporal context to V3 baseline (40.91% IoU) → Target 47-50% IoU

## Current Problem
V3 only sees current timestep `obs_t` → no memory of fire history

## Solution
Process last N timesteps with LSTM → capture fire spread velocity/acceleration

---

## Architecture

### Input: Temporal Observation Sequence
```python
# Current (V3): single timestep
obs_t: (B, 14, H, W)

# New (V7): last 3 timesteps
obs_seq: (B, 3, 14, H, W)  # [t-2, t-1, t]
```

### Model Components

#### 1. CNN Encoder (Per-Timestep)
```python
class CNNEncoder(nn.Module):
    """Encode spatial features for each timestep independently"""
    def __init__(self):
        self.conv1 = nn.Conv2d(14, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

    def forward(self, x):
        # x: (B, 14, H, W)
        # output: (B, 128, H, W)
```

**Process sequence:**
```python
# Apply CNN to each timestep
feature_seq = []
for t in range(3):
    feat_t = cnn_encoder(obs_seq[:, t])  # (B, 128, H, W)
    feature_seq.append(feat_t)
# feature_seq: list of 3 x (B, 128, H, W)
```

#### 2. Temporal LSTM (Pixel-wise)
```python
class TemporalLSTM(nn.Module):
    """Process temporal sequence at each spatial location"""
    def __init__(self, feature_dim=128, hidden_dim=128):
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

    def forward(self, feature_seq):
        # feature_seq: (B, T, C, H, W) = (B, 3, 128, H, W)
        B, T, C, H, W = feature_seq.shape

        # Reshape: (B, H, W, T, C)
        x = feature_seq.permute(0, 3, 4, 1, 2)
        x = x.reshape(B * H * W, T, C)

        # LSTM: process temporal dimension
        lstm_out, _ = self.lstm(x)  # (B*H*W, T, hidden_dim)

        # Take last timestep output
        temporal_features = lstm_out[:, -1, :]  # (B*H*W, hidden_dim)

        # Reshape back: (B, hidden_dim, H, W)
        temporal_features = temporal_features.reshape(B, H, W, -1)
        temporal_features = temporal_features.permute(0, 3, 1, 2)

        return temporal_features  # (B, 128, H, W)
```

#### 3. Policy Head (Same as V3)
```python
# Extract 3x3 features around each burning cell
# FC: 1152 → 256 → 64 → 8 (8-neighbor predictions)
```

#### 4. Value Head (Same as V3)
```python
# Global average pooling + FC
# Output: scalar state value
```

---

## Full Forward Pass

```python
class A3C_TemporalModel(nn.Module):
    def __init__(self):
        self.cnn_encoder = CNNEncoder()
        self.temporal_lstm = TemporalLSTM(128, 128)
        self.policy_head = PolicyHead()
        self.value_head = ValueHead()

    def forward(self, obs_seq, burning_cells):
        # obs_seq: (B, 3, 14, H, W)
        B, T, C, H, W = obs_seq.shape

        # Step 1: CNN encode each timestep
        feature_seq = []
        for t in range(T):
            feat_t = self.cnn_encoder(obs_seq[:, t])  # (B, 128, H, W)
            feature_seq.append(feat_t)
        feature_seq = torch.stack(feature_seq, dim=1)  # (B, T, 128, H, W)

        # Step 2: Temporal LSTM
        temporal_features = self.temporal_lstm(feature_seq)  # (B, 128, H, W)

        # Step 3: Policy head (per burning cell)
        actions, log_probs = self.policy_head(temporal_features, burning_cells)

        # Step 4: Value head
        state_value = self.value_head(temporal_features)

        return actions, log_probs, state_value
```

---

## Environment Modifications

### Current: Single Timestep
```python
obs = env._get_obs(t)  # (14, H, W)
```

### New: Temporal Window
```python
class WildfireEnvTemporal:
    def _get_obs_sequence(self, t, window_size=3):
        """Get last window_size timesteps"""
        obs_list = []
        for i in range(window_size):
            t_i = max(0, t - (window_size - 1 - i))
            obs_i = self._get_obs(t_i)
            obs_list.append(obs_i)
        return np.stack(obs_list, axis=0)  # (3, 14, H, W)

    def reset(self):
        self.t = 0
        obs_seq = self._get_obs_sequence(0, window_size=3)
        return obs_seq, {'t': 0}

    def step(self, predicted_burn_mask):
        # ... compute reward ...
        self.t += 1
        obs_seq = self._get_obs_sequence(self.t, window_size=3)
        return obs_seq, reward, done, info
```

**Padding for early timesteps:**
```python
# t=0: [obs_0, obs_0, obs_0]  # Repeat first observation
# t=1: [obs_0, obs_0, obs_1]
# t=2: [obs_0, obs_1, obs_2]
# t=3: [obs_1, obs_2, obs_3]
```

---

## Training Modifications

### Worker Changes
```python
# Old: single obs
obs = env.reset()

# New: obs sequence
obs_seq = env.reset()  # (3, 14, H, W)

# Forward pass
obs_seq_tensor = torch.FloatTensor(obs_seq).unsqueeze(0)  # (1, 3, 14, H, W)
actions, log_probs, value = model(obs_seq_tensor, burning_cells)
```

### Memory Considerations
- **V3**: 1 timestep × 14 channels = 14 feature maps
- **V7**: 3 timesteps × 14 channels = 42 feature maps (3x memory)
- **With 4 workers**: Should still fit in 64GB RAM (V3 uses ~40%)
- **GPU Memory**: 3 timesteps × batch_size should fit in 12GB easily

---

## Hyperparameters

### From V3 (Keep)
```python
num_workers = 4
lr = 7e-5
entropy_coef = 0.015
min_episode_length = 4
gamma = 0.99
value_loss_coef = 0.5
max_grad_norm = 0.5
```

### New for V7
```python
temporal_window_size = 3  # Look back 3 timesteps
lstm_hidden_dim = 128  # Match CNN output
lstm_num_layers = 2
lstm_dropout = 0.1
```

### Training Setup
```python
max_episodes = 1000  # Same as V3
checkpoint_every = 50
eval_every = 100
```

---

## Expected Benefits

### What LSTM Should Learn:
1. **Fire spread velocity**: "Fire moved 2 cells in last 2 timesteps → predict 2-3 cells next"
2. **Acceleration patterns**: "Fire is speeding up → predict more aggressive spread"
3. **Wind dynamics**: "Wind shifted from NW to N → adjust spread direction"
4. **Temporal weather trends**: "Humidity dropping for 3 timesteps → fire intensifying"

### Failure Modes to Avoid:
1. **Overfitting to noise**: Use dropout in LSTM
2. **Vanishing gradients**: 2-layer LSTM should be okay for window=3
3. **Memory explosion**: 3x memory usage, monitor carefully

---

## Implementation Steps

### Step 1: Modify Environment (1 day)
- Add `_get_obs_sequence()` method
- Update `reset()` and `step()` to return sequences
- Test with dummy forward passes

### Step 2: Build Model (1 day)
- Implement `TemporalLSTM` module
- Integrate with V3's CNN encoder and heads
- Test forward/backward pass shapes

### Step 3: Update Worker (0.5 day)
- Handle obs_seq instead of obs
- Update tensor shapes throughout

### Step 4: Training (2-3 days)
- Train for 1000 episodes
- Monitor for memory issues
- Compare with V3 baseline

### Step 5: Evaluation (0.5 day)
- Run validation set evaluation
- Target: 47-50% IoU (from V3's 40.91%)

---

## GPU Utilization Strategy

### Hybrid CPU-GPU A3C
```python
# Global model on GPU
global_model = A3C_TemporalModel().cuda()

# Workers on CPU (environment stepping)
workers = []
for i in range(4):
    worker = Worker(global_model, env, rank=i, device='cpu')
    # Worker copies model to CPU for local rollouts
    # Sends gradients back to GPU for updates
    workers.append(worker)
```

### Benefits:
1. **GPU**: Fast LSTM forward/backward during gradient updates
2. **CPU**: Parallel environment stepping (4 workers)
3. **No bottleneck**: Model updates on GPU while workers step environments

### Memory Distribution:
- **GPU (12GB)**: Global model (~500MB) + gradient buffers (~1GB) = plenty of space
- **CPU (64GB)**: 4 workers × 10GB each = 40GB (safe)

---

## Alternative: Lighter Temporal Model

If LSTM is too heavy, try **ConvLSTM** or **Temporal Convolution**:

### Option A: ConvLSTM
```python
# Spatial LSTM (more efficient)
self.conv_lstm = ConvLSTM(
    input_dim=128,
    hidden_dim=128,
    kernel_size=3,
    num_layers=2
)
```

### Option B: Temporal Convolution (1D conv over time)
```python
# Simpler: 1D conv over temporal dimension
self.temporal_conv = nn.Conv1d(
    in_channels=128,
    out_channels=128,
    kernel_size=3,  # Process 3 timesteps
    padding=1
)
```

Start with standard LSTM, fall back to these if memory is an issue.

---

## Success Criteria

### Training Metrics (1000 episodes)
- **Best IoU**: > 47% (target: 47-50%)
- **Improvement over V3**: +6-9% IoU
- **Training stability**: Loss stays in [-10, +5] range
- **Memory usage**: < 60% RAM with 4 workers

### Validation Metrics
- **Mean IoU**: > 40% (population-level)
- **Median IoU**: > 35%
- **Best episode IoU**: > 60%

### Qualitative Improvements
- Better prediction on fast-spreading fires
- Accurate modeling of wind-driven spread
- Reduced false positives in slow-spread scenarios

---

## Fallback Plan

If V7 doesn't reach 47% IoU:

1. **Debug**: Check if LSTM is actually learning temporal patterns
   - Visualize LSTM hidden states
   - Ablation: compare V7 vs V3 on same episodes
2. **Tune window size**: Try 2, 4, or 5 timesteps
3. **Simplify**: Try temporal convolution instead of LSTM
4. **Combine**: Add spatial attention to V7 (V7.5)

---

## Files to Create

1. `rl_training/a3c/model_v7.py` - Temporal model architecture
2. `rl_training/a3c/worker_v7.py` - Worker with temporal observations
3. `rl_training/a3c/train_v7.py` - Training script
4. `rl_training/wildfire_env_temporal_v2.py` - Environment with obs sequences

---

**Target Completion**: 7 days (Dec 22)
**Expected Result**: 47-50% IoU → +6-9% improvement over V3
**Risk Level**: Medium (temporal modeling is well-studied)

---

## ⚠️ POSTMORTEM: V7 FAILED CATASTROPHICALLY

**Date**: 2025-11-16
**Result**: 8.13% IoU at episode 560
**Comparison**: -80% degradation vs V3's 40.91% IoU

### What Was Actually Implemented (vs Plan)

| Aspect | Original Plan | What Was Built | Impact |
|--------|---------------|----------------|--------|
| Temporal Model | 2-layer LSTM (128 hidden) | 3D Convolution | ❌ CRITICAL |
| CNN Encoder | Keep V3 (32→64→128) | Reduced (32→64) | ❌ CRITICAL |
| Window Size | 3 timesteps | 3 timesteps | ⚠️ Too small |
| Params | ~480K | ~450K | Lighter but worse |
| Training | 1000 episodes, 4 workers | 5000 episodes, 4 workers | Same |

### Critical Mistakes

#### Mistake #1: Used 3D Conv Instead of LSTM
**Decision**: "3D Conv is MUCH lighter than LSTM!"
**Rationale**: Save memory and computation
**Why It Failed**:
- **3D convolutions**: Capture local spatial-temporal patterns in receptive field
- **LSTMs**: Maintain hidden STATE to remember temporal dynamics
- **Fire spread needs**: Velocity, acceleration, temporal trends → needs STATE
- **Result**: Model couldn't learn fire spread velocity or acceleration

**Code Evidence** (model_v7.py:44-50):
```python
# Temporal 3D Conv (MUCH lighter than LSTM!)
self.temporal_conv = nn.Sequential(
    nn.Conv3d(64, 96, kernel_size=(3, 3, 3), padding=(0, 1, 1)),
    nn.ReLU(),
    nn.Conv3d(96, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
    nn.ReLU(),
)
```

#### Mistake #2: Reduced CNN Encoder Capacity
**Decision**: Cut 128-channel layer, keep only 32→64
**Rationale**: "3D conv will compensate for weaker encoder"
**Why It Failed**:
- V3's 128-channel layer was CRITICAL for representation power
- Reducing to 64 channels lost spatial feature quality
- 3D conv couldn't compensate for this loss
- Trading spatial features for temporal modeling was wrong approach

**Code Evidence** (model_v7.py:35-41):
```python
# CNN encoder (LIGHTER - 64 channels instead of 128)
self.encoder = nn.Sequential(
    nn.Conv2d(in_channels, 32, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, 3, padding=1),  # Stopped here, no 128!
    nn.ReLU(),
)
```

#### Mistake #3: Window Size Too Small
**Plan**: 3 timesteps
**Problem**: Not enough temporal context to learn velocity/acceleration
**Should Have Been**: 5-7 timesteps minimum

### Performance Analysis

| Metric | V3 Baseline | V7 Result | Change |
|--------|-------------|-----------|--------|
| Best IoU | 40.91% | 8.13% | **-80.1%** |
| Episode | 731 | 560 | -23% |
| Architecture | Simple, proven | Complex, wrong | - |
| Params | 417K | ~450K | +8% |

**Training Dynamics**:
- Model struggled from the start
- Never exceeded ~10% IoU
- Worse than V5 (12% F1), V6 (36.36% IoU)
- Even worse than min-len 2 filtering (9.4% IoU)
- Comparable to catastrophically bad early V3 experiments

### Root Cause Analysis

**Fundamental Misunderstanding**: Optimized for "lighter" instead of "better"

1. **3D Conv Philosophy Error**:
   - 3D convs are great for: Action recognition, video classification (pattern matching)
   - 3D convs are BAD for: Sequential prediction with velocity/acceleration (state needed)
   - Fire spread is the latter: needs to remember how fast fire is moving

2. **Architecture Trade-off Error**:
   - Thought: "Reduce spatial quality, add temporal modeling"
   - Reality: Need BOTH spatial quality AND temporal memory
   - Should have: Keep V3 encoder + add LSTM

3. **Memory Optimization Error**:
   - Fear: LSTM would use too much memory
   - Reality: ~130K params for 2-layer LSTM is negligible
   - Wrong priority: Saved 30KB, lost 80% performance

### Correct V7.5 Architecture

**Philosophy**: Performance first, then optimize

```python
class A3C_TemporalModel_V7_5(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. KEEP FULL V3 ENCODER (128 channels)
        self.encoder = nn.Sequential(
            nn.Conv2d(14, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),  # DON'T CUT THIS!
            nn.ReLU(),
        )

        # 2. USE LSTM FOR TEMPORAL STATE
        self.temporal_lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # 3. ADD LAYER NORM FOR STABILITY
        self.layer_norm = nn.LayerNorm(128)

        # 4. Keep V3's policy and value heads
        self.policy_head = V3_PolicyHead()
        self.value_head = V3_ValueHead()

    def forward(self, obs_seq):
        # obs_seq: (B, T, 14, H, W) where T=5
        B, T, C, H, W = obs_seq.shape

        # Encode each timestep
        features = []
        for t in range(T):
            feat_t = self.encoder(obs_seq[:, t])  # (B, 128, H, W)
            features.append(feat_t)
        features = torch.stack(features, dim=1)  # (B, T, 128, H, W)

        # Process with LSTM at each spatial location
        # Reshape: (B, H, W, T, 128)
        x = features.permute(0, 3, 4, 1, 2)
        x = x.reshape(B * H * W, T, 128)

        # LSTM
        lstm_out, _ = self.temporal_lstm(x)
        temporal_features = lstm_out[:, -1, :]  # Last timestep

        # Layer norm
        temporal_features = self.layer_norm(temporal_features)

        # Reshape back
        temporal_features = temporal_features.reshape(B, H, W, 128)
        temporal_features = temporal_features.permute(0, 3, 1, 2)

        # Policy and value heads (same as V3)
        return temporal_features
```

**Key Differences from V7**:
1. ✅ Full V3 encoder (128 channels)
2. ✅ LSTM instead of 3D conv
3. ✅ Increase window to 5 timesteps
4. ✅ Layer norm for stability
5. ✅ ~480K params (worth it!)

### Lessons Learned

1. **Architectural Choice Matters More Than Efficiency**
   - Wrong architecture = catastrophic failure
   - Right architecture = worth the extra 30K params

2. **Temporal Modeling Requires State**
   - 3D conv: Pattern matching in temporal window
   - LSTM/GRU: State memory for velocity/acceleration
   - Fire spread needs the latter

3. **Don't Sacrifice Proven Components**
   - V3's 128-channel encoder was proven critical
   - Reducing it was penny-wise, pound-foolish

4. **Test Assumptions Early**
   - Should have tested 3D conv on small dataset first
   - Would have caught this after 50-100 episodes
   - Instead wasted 5000 episodes (~1 hour compute)

5. **Performance > Efficiency (at research stage)**
   - Optimize for correctness first
   - Optimize for efficiency later
   - Can't optimize a broken architecture

### Files & Code Location

- **Model**: `rl_training/a3c/model_v7.py` (FAILED implementation)
- **Training**: `rl_training/a3c/train_v7.py`
- **Worker**: `rl_training/a3c/worker_v7.py`
- **Checkpoint**: None saved (performance never exceeded 10% IoU)
- **WandB Run**: `run-20251115_233004-h5rt3iop` (a3c-v7-workers4)

### Next Steps

**DO NOT:**
- ❌ Try to "fix" V7 by tuning hyperparameters
- ❌ Use 3D conv for any temporal modeling
- ❌ Reduce encoder capacity for "efficiency"

**DO:**
- ✅ Implement V7.5 with correct LSTM architecture
- ✅ Keep full V3 encoder (128 channels)
- ✅ Use 5 timestep window (not 3)
- ✅ Add layer norm for training stability
- ✅ Test on 100 episodes before full training

---

**Postmortem Written**: 2025-11-16
**Next After V7 Failure**: V7.5 with correct LSTM architecture → Target 50-55% IoU
