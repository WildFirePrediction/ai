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

**Next After V7**: Spatial attention (Week 2) → 52-58% IoU
