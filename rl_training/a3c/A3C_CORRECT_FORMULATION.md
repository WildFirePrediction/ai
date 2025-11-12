# A3C Correct Problem Formulation for Wildfire Prediction

## CRITICAL: What We're Actually Trying to Solve

### ❌ WRONG (Current Implementation)
**Problem**: Predict which of ALL 30,000 cells in the grid will burn next
**Action Space**: 30,000-dimensional Bernoulli (independent prediction per cell)
**Why It Fails**:
- Massive action space → log probability explodes to -200,000
- No structure → can't learn spatial fire spread patterns
- Credit assignment impossible across 30,000 decisions

### ✅ CORRECT (Target Implementation)
**Problem**: For each CURRENTLY BURNING CELL, predict which of its 8 neighbors will burn next
**Action Space**: K burning cells × 8 neighbors (typically 10-100 total decisions)
**Why It Works**:
- Small, structured action space
- Directly models fire spread mechanism (cell-to-cell)
- Clear credit assignment (did this burning cell spread to neighbor?)
- Matches MCTS-A3C approach

---

## Problem Formulation Details

### State (Input)
**Environmental features at timestep t**: `(14, H, W)` tensor
- Static features: elevation, slope, aspect, land cover, fuel model (5 channels)
- Fire features at t: fire mask, intensity, temperature, age (4 channels)
- Weather at t: temperature, humidity, wind speed/direction, pressure (5 channels)

**Current fire mask at t**: `(H, W)` binary mask
- 1 = burning cell
- 0 = not burning

### Action (Output)
**For EACH burning cell at position (i, j)**:
- Predict 8-neighbor spread: **8-dimensional Bernoulli vector**
- Neighbors: N, NE, E, SE, S, SW, W, NW (in that order)
- Example: `[1, 0, 1, 0, 0, 0, 0, 1]` = fire spreads North, East, NW

**Total action per timestep**:
- If K cells are burning at timestep t
- Action = List of K × 8-dim vectors
- Example: 10 burning cells → 80 binary decisions

### Target (Ground Truth)
**Actual new burns at timestep t+1**:
```python
burning_at_t = (fire_mask[t] > 0)
burning_at_t1 = (fire_mask[t+1] > 0)
new_burns = burning_at_t1 & ~burning_at_t  # Cells that just started burning
```

**Per-cell ground truth**:
For each burning cell at (i, j), check which of its 8 neighbors are in `new_burns`:
```python
neighbors = [(i-1,j), (i-1,j+1), (i,j+1), (i+1,j+1),
             (i+1,j), (i+1,j-1), (i,j-1), (i-1,j-1)]

target_8d = [1 if neighbor in new_burns else 0 for neighbor in neighbors]
```

### Reward (Dense, Per-Timestep)
**CRITICAL: Reward at EVERY timestep, not just episode end!**

For each timestep, compute IoU between predicted and actual new burns:
```python
# At timestep t, agent predicts which cells burn at t+1
predicted_burns = gather_predictions_from_burning_cells()  # (H, W) binary mask
actual_new_burns = get_new_burns(t, t+1)  # (H, W) binary mask

intersection = (predicted_burns & actual_new_burns).sum()
union = (predicted_burns | actual_new_burns).sum()

reward_t = intersection / (union + 1e-8)  # IoU reward, range [0, 1]
```

**Why dense rewards**:
- Agent gets feedback immediately
- Learns which cell-to-neighbor spread patterns are correct
- No need to wait until episode end
- Much stronger learning signal

---

## Model Architecture

### High-Level Structure
```
Input: (B, 14, H, W) environmental features
       + (B, H, W) current fire mask

↓
Shared CNN Encoder
↓
For each burning cell (i, j):
    ├─→ Extract local features around (i, j)
    ├─→ Policy head: predict 8-neighbor spread probabilities
    └─→ Value head: estimate value of this cell's state
```

### Detailed Architecture

#### 1. Shared CNN Encoder
```python
input: (B, 14, H, W)
↓
Conv2d(14 → 32, k=3, p=1) + ReLU
Conv2d(32 → 64, k=3, p=1) + ReLU
Conv2d(64 → 128, k=3, p=1) + ReLU
↓
output: (B, 128, H, W) feature map
```

#### 2. Extract Burning Cell Locations
```python
fire_mask = input[:, fire_channel_idx]  # (B, H, W)
burning_indices = torch.nonzero(fire_mask > 0)  # List of (batch, i, j)
```

#### 3. Per-Cell Feature Extraction
For each burning cell at (i, j):
```python
# Extract 3x3 local features
local_features = features[:, :, i-1:i+2, j-1:j+2]  # (B, 128, 3, 3)
local_flat = local_features.flatten(1)  # (B, 128*9)
```

#### 4. Policy Head (Per Cell)
```python
input: (B, 128*9) local features
↓
Linear(128*9 → 256) + ReLU
Linear(256 → 8)  # 8 neighbors
↓
output: (B, 8) logits for 8-neighbor spread
```

#### 5. Value Head (Global)
```python
input: (B, 128, H, W) full feature map
↓
AdaptiveAvgPool2d(1) → (B, 128, 1, 1)
Flatten → (B, 128)
Linear(128 → 64) + ReLU
Linear(64 → 1)
↓
output: (B, 1) state value
```

---

## Action Sampling and Execution

### During Training (Worker)
```python
# 1. Get feature map from encoder
features = model.encoder(obs)  # (1, 128, H, W)

# 2. Find all burning cells
burning_cells = torch.nonzero(fire_mask > 0)  # [(i, j), (i', j'), ...]

# 3. For each burning cell, get 8-neighbor predictions
all_actions = []
all_log_probs = []

for (i, j) in burning_cells:
    local_features = extract_local(features, i, j)
    logits_8d = model.policy_head(local_features)  # (8,)
    probs_8d = torch.sigmoid(logits_8d)

    # Sample 8-dimensional action
    action_8d = torch.bernoulli(probs_8d)  # (8,) binary

    # Compute log prob
    log_prob = (action_8d * torch.log(probs_8d + 1e-8) +
                (1 - action_8d) * torch.log(1 - probs_8d + 1e-8)).sum()

    all_actions.append((i, j, action_8d))
    all_log_probs.append(log_prob)

# 4. Convert actions to grid prediction
predicted_grid = torch.zeros(H, W)
for (i, j, action_8d) in all_actions:
    neighbors = get_8_neighbors(i, j)
    for n_idx, (ni, nj) in enumerate(neighbors):
        if action_8d[n_idx] == 1:
            predicted_grid[ni, nj] = 1

# 5. Step environment with predicted_grid
next_obs, reward, done, info = env.step(predicted_grid)
```

### Reward Computation (in Environment)
```python
def step(self, predicted_burn_mask):
    """
    Args:
        predicted_burn_mask: (H, W) binary prediction of next burns

    Returns:
        reward: IoU between predicted and actual new burns
    """
    # Get actual new burns at t+1
    actual_mask_t = self.fire_masks[self.t] > 0
    actual_mask_t1 = self.fire_masks[self.t + 1] > 0
    new_burns = actual_mask_t1 & ~actual_mask_t

    # Compute IoU
    intersection = (predicted_burn_mask & new_burns).sum()
    union = (predicted_burn_mask | new_burns).sum()

    reward = float(intersection / (union + 1e-8))

    self.t += 1
    done = (self.t >= self.T - 1)

    return self._get_obs(self.t), reward, done, {'t': self.t}
```

---

## A3C Training Flow

### Worker Process
```python
while not done:
    # 1. Sync with shared model
    local_model.load_state_dict(shared_model.state_dict())

    # 2. Collect trajectory with DENSE rewards
    states, actions, rewards = [], [], []

    for step in range(max_steps_per_episode):
        # Get action from policy
        action, log_prob, value = local_model.get_action(state)

        # Step environment - REWARD AT EVERY STEP!
        next_state, reward, done, info = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)  # Dense reward!

        if done:
            break

        state = next_state

    # 3. Compute returns (discounted cumulative rewards)
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    # 4. Compute losses
    advantages = returns - values
    policy_loss = -(log_probs * advantages).mean()
    value_loss = (values - returns).pow(2).mean()

    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    # 5. Backprop and update shared model
    total_loss.backward()
    clip_grad_norm_(local_model.parameters(), max_grad_norm)

    with lock:
        copy_gradients(local_model → shared_model)
        optimizer.step()
```

---

## Key Differences from Current (Wrong) Implementation

| Aspect | Wrong (Full Grid) | Correct (Per-Cell 8-Neighbor) |
|--------|-------------------|--------------------------------|
| **Action space** | 30,000-dim | K × 8-dim (10-100 typically) |
| **Action structure** | Independent per cell | Structured per burning cell |
| **Log probability** | Sum over 30,000 | Sum over K × 8 (~80) |
| **Reward** | Sparse (end of episode) | Dense (every timestep) |
| **Credit assignment** | Impossible | Clear (per burning cell) |
| **Matches fire physics** | No | Yes (cell-to-cell spread) |

---

## Expected Performance

With correct formulation:
- **Action space**: ~10-100 decisions per timestep (tractable)
- **Log probs**: ~-10 to -100 (reasonable, not -200,000)
- **Dense rewards**: 0.0-0.5 IoU per timestep (informative gradient)
- **Target IoU**: 0.15-0.30 after 1000 episodes (comparable to supervised)

---

## Implementation Checklist

### Model (`model_v2.py`)
- [ ] Shared CNN encoder: (14, H, W) → (128, H, W)
- [ ] Function to extract burning cell locations from fire mask
- [ ] Function to extract local 3×3 features around cell (i, j)
- [ ] Policy head: local features → 8-dim logits
- [ ] Value head: global features → scalar value
- [ ] `get_action_and_value()`: returns per-cell actions, log_probs, entropy, value

### Worker (`worker_v3.py`)
- [ ] Find all burning cells at timestep t
- [ ] For each burning cell, sample 8-neighbor action
- [ ] Aggregate into (H, W) prediction grid
- [ ] Step environment with prediction
- [ ] Collect DENSE reward at every timestep
- [ ] Store trajectory: (states, actions, rewards)
- [ ] Compute returns with discount factor
- [ ] Compute policy and value losses
- [ ] Update shared model

### Environment Modification (`wildfire_env_spatial.py`)
- [ ] `step()` returns dense reward (IoU at this timestep)
- [ ] NOT sparse reward at episode end

### Training (`train_v3.py`)
- [ ] Use filtered episodes (keep from V2)
- [ ] Workers sample from filtered episodes
- [ ] Log per-timestep rewards (not just episode total)
- [ ] Monitor IoU per timestep

---

## Why This Matches MCTS-A3C

**MCTS-A3C approach**:
- Uses Monte Carlo Tree Search to plan ahead
- Each node in tree = one burning cell's state
- Actions = 8-neighbor spread decisions
- A3C provides policy prior for MCTS

**Our formulation**:
- Same action structure (8-neighbor per burning cell)
- Can plug into MCTS later for planning
- A3C learns good spread patterns first
- Then MCTS uses A3C policy to plan multi-step

---

## Success Criteria

After 1000 episodes, we should see:
- **Average reward per timestep**: 0.1-0.3 (10-30% IoU)
- **Episode total reward**: 1.0-3.0 (sum of 10-20 timesteps)
- **Loss**: Stable around 1.0-5.0 (not -200,000!)
- **Learning curve**: Steady improvement over episodes

If these aren't met, the formulation is still wrong.

---

## References

- Fire spread is inherently cell-to-cell (physics-based)
- MCTS-A3C papers use per-cell action spaces for spatial problems
- Dense rewards > sparse rewards for RL (proven in literature)
- 8-neighbor Moore neighborhood is standard for cellular automata

---

**BOTTOM LINE**: Predict 8-neighbor spread per burning cell with dense rewards, NOT full grid with sparse rewards!
