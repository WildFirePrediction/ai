# Wildfire Prediction RL Training Pipeline - Master Plan

## Overview
This document outlines the complete process from raw data to trained RL model for wildfire prediction in South Korea.

---

## Phase 1: Data Embedding (COMPLETE)
**Status**: Scripts ready in `embedding_src/`  
**Output Directory**: `embedded_data/`

### 1.1 Individual Data Source Embedding
Sequential processing of each data source to normalized, aligned spatial grids:

1. **NASA VIIRS Fire Data** (`01_nasa_viirs_embedding.py`)
   - Input: `data/NASA/VIIRS/*/clustered_fire_archive_*.csv`
   - Output: `embedded_data/nasa_viirs_embedded.parquet`
   - Features: 
     - Fire location (x, y in EPSG:5179)
     - Temperature (te): Brightness temp in Celsius
     - Intensity (i): Log-scaled combination of FRP + Brightness
     - Temporal (tm): Elapsed time in fire episode
     - Episode IDs: Pre-clustered fire instances
   - Record format: One row per fire detection with episode_id

2. **DEM & RSP** (`02_dem_rsp_embedding.py`)
   - Input: `data/DigitalElevationModel/90m_GRS80.tif`
   - Input: `data/RelativeSlopePosition/`
   - Output: `embedded_data/dem_rsp_embedded.tif` (2-band raster)
   - Features:
     - Band 1: Normalized elevation (0-1)
     - Band 2: Normalized RSP (0-1)
   - Grid: Resampled to common resolution

3. **Land Cover Map** (`03_lcm_embedding.py`)
   - Input: `data/LandCoverMap/SG05_*/` (shapefiles)
   - Output: `embedded_data/lcm_embedded.tif` (class indices)
   - Output: `embedded_data/lcm_class_mapping.json`
   - Features: Land cover type classification (residential, forest, agricultural, etc.)

4. **Forest Stand Map** (`04_fsm_embedding.py`)
   - Input: `data/ForestStandMap/*/` (shapefiles by province)
   - Output: `embedded_data/fsm_embedded.tif` (class indices)
   - Output: `embedded_data/fsm_class_mapping.json`
   - Features: Forest type, age, density classifications

5. **NDVI** (`05_ndvi_embedding.py`)
   - Input: `data/NDVI/MOD13Q1.061_250m_aid0001.nc` (NetCDF)
   - Output: `embedded_data/ndvi_embedded.tif` (multi-band temporal)
   - Features: Vegetation health index over time

6. **KMA Weather** (`06_kma_weather_embedding.py`)
   - Input: `data/KMA/*/AWS_*.csv`
   - Output: `embedded_data/kma_weather_embedded.tif` (5-band raster)
   - Features:
     - Band 1: Wind speed (w)
     - Band 2-3: Wind direction components (d_x, d_y)
     - Band 4: Relative humidity (rh)
     - Band 5: Precipitation (r)
   - Method: Station data interpolated to grid

### 1.2 Final State Composition
**Script**: `07_final_state_composition.py`

**Outputs**:
- `embedded_data/state_base.npz`:
  - `continuous`: (C, H, W) float32 array - all continuous features stacked
  - `lcm_classes`: (H, W) uint16 - land cover class indices
  - `fsm_classes`: (H, W) uint16 - forest stand class indices
  - Feature names and metadata

- `embedded_data/fire_episodes.parquet`:
  - Episode metadata with bounding boxes
  - Start/end times
  - Fire statistics per episode

- `embedded_data/metadata.json`:
  - Grid dimensions (H, W)
  - Spatial reference (EPSG:5179)
  - Transform matrix
  - Feature channels and names
  - Class mappings

**State Vector Structure**:
```
Continuous features (8 channels):
[0] dem_norm      - Normalized elevation
[1] rsp_norm      - Relative slope position
[2] ndvi_norm     - Vegetation index
[3] w_norm        - Wind speed
[4] d_x_norm      - Wind direction X
[5] d_y_norm      - Wind direction Y
[6] rh_norm       - Humidity
[7] r_norm        - Precipitation

Categorical features (embedded separately):
- lcm_classes     - Land cover type (one-hot or embedding)
- fsm_classes     - Forest stand type (one-hot or embedding)
```

---

## Phase 2: Environment Tiling (NEXT)
**Status**: To be implemented in `tilling_src/`  
**Purpose**: Create RL training environments from embedded data

### 2.1 Spatial Tiling
**Script**: `tilling_src/01_spatial_tiling.py`

**Objective**: Divide the large Korean peninsula grid into manageable tiles for RL training

**Input**:
- `embedded_data/state_base.npz` - Full country grid
- `embedded_data/fire_episodes.parquet` - Fire locations
- `embedded_data/metadata.json` - Grid specifications

**Process**:
1. Define tile size (e.g., 256x256 or 512x512 pixels)
2. Create overlapping tiles with stride (e.g., stride=128 for 50% overlap)
3. Filter tiles:
   - Must contain at least one fire episode
   - Must have sufficient valid data (non-null pixels)
   - Optional: Ensure diversity of terrain/land cover types
4. Extract tile boundaries in grid coordinates
5. Assign fire episodes to tiles based on spatial intersection

**Output**: `tilled_data/tile_index.parquet`
```
Columns:
- tile_id: Unique identifier
- x_start, x_end, y_start, y_end: Grid coordinates
- lon_min, lon_max, lat_min, lat_max: Geographic bounds
- num_fire_episodes: Count of fires in tile
- fire_episode_ids: List of episode IDs
- land_cover_diversity: Shannon entropy of land types
- forest_coverage: Percentage of forested area
- avg_elevation: Mean elevation in tile
```

### 2.2 Temporal Segmentation
**Script**: `tilling_src/02_temporal_segmentation.py`

**Objective**: Create temporal windows for each fire episode in each tile

**Input**:
- `tilled_data/tile_index.parquet`
- `embedded_data/fire_episodes.parquet`
- `embedded_data/nasa_viirs_embedded.parquet`

**Process**:
1. For each tile and each fire episode:
   - Extract fire detection timestamps
   - Define temporal window (e.g., -24h before first detection to +7 days after)
   - Identify all fire progression snapshots within window
2. Create timestep sequence:
   - t=0: Initial fire detection
   - t=1,2,3...: Subsequent detections at regular intervals (e.g., 6h)
3. For each timestep, extract:
   - Fire mask: (H, W) binary indicating fire presence
   - Fire intensity: (H, W) float with intensity values
   - Fire age: (H, W) float with time since ignition

**Output**: `tilled_data/fire_sequences/`
```
Structure:
- tile_{tile_id}_episode_{ep_id}.npz:
  - timestamps: Array of datetime objects
  - fire_masks: (T, H, W) binary arrays
  - fire_intensity: (T, H, W) float arrays
  - fire_age: (T, H, W) float arrays
  - T: Number of timesteps
```

### 2.3 Environment Generation
**Script**: `tilling_src/03_environment_generation.py`

**Objective**: Combine spatial tiles, temporal sequences, and static features into RL-ready environments

**Input**:
- `tilled_data/tile_index.parquet`
- `tilled_data/fire_sequences/`
- `embedded_data/state_base.npz`

**Process**:
1. For each (tile, episode) pair:
   - Load tile region from state_base
   - Load fire sequence for episode
   - Combine into environment state
2. Create state representation at each timestep:
   ```python
   state = {
       'static_continuous': (8, H, W),    # DEM, RSP, NDVI, etc.
       'static_categorical': {             # Land cover, forest type
           'lcm': (H, W),
           'fsm': (H, W)
       },
       'dynamic_fire': {                   # Fire state at time t
           'mask': (H, W),
           'intensity': (H, W),
           'age': (H, W)
       },
       'previous_actions': (H, W)         # Suppression actions taken so far
   }
   ```
3. Define action space:
   - Spatial: Select grid cell(s) for suppression
   - Action types: Water drop, fire line, controlled burn, etc.
4. Define reward function components:
   - Negative: Fire spread area, intensity growth, property damage
   - Positive: Fire contained, suppression success
   - Costs: Suppression resource usage

**Output**: `tilled_data/environments/`
```
Structure:
- env_{tile_id}_{episode_id}.pkl: Serialized environment object
- env_manifest.json: List of all environments with metadata
```

### 2.4 Dataset Split & Validation
**Script**: `tilling_src/04_dataset_split.py`

**Objective**: Split environments into train/validation/test sets

**Input**:
- `tilled_data/environments/env_manifest.json`

**Process**:
1. Stratified split by:
   - Geographic region (to test generalization)
   - Fire season (to test temporal robustness)
   - Fire size (to ensure diversity)
2. Typical split: 70% train, 15% validation, 15% test
3. Ensure no spatial overlap between sets (avoid data leakage)

**Output**: `tilled_data/splits/`
```
- train_envs.json
- val_envs.json
- test_envs.json
```

---

## Phase 3: RL Training Infrastructure (FUTURE)
**Directory**: `rl_training/`

### 3.1 Environment Wrapper
**File**: `rl_training/wildfire_env.py`

**Implement Gym-style environment**:
```python
class WildfireEnv(gym.Env):
    def __init__(self, env_data_path):
        # Load environment from tilled_data
        self.state_shape = ...
        self.action_space = ...
        self.observation_space = ...
    
    def reset(self):
        # Initialize episode at t=0
        return initial_state
    
    def step(self, action):
        # Apply suppression action
        # Simulate fire spread (or use actual next timestep)
        # Calculate reward
        return next_state, reward, done, info
    
    def render(self):
        # Visualize current state
```

### 3.2 Neural Network Architecture
**File**: `rl_training/models.py`

**Options**:

1. **CNN-based Policy**:
   - Input: Multi-channel state (static + dynamic)
   - Encoder: ResNet or U-Net style
   - Categorical embedding for LCM/FSM
   - Output: Action logits per cell

2. **Transformer-based Policy**:
   - Treat grid as sequence of patches
   - Self-attention across spatial locations
   - Better for long-range dependencies

3. **Graph Neural Network**:
   - Represent fire spread as graph
   - Nodes: Grid cells or fire clusters
   - Edges: Spatial adjacency and wind connectivity

### 3.3 Training Algorithm
**File**: `rl_training/train.py`

**Recommended Algorithms**:

1. **PPO (Proximal Policy Optimization)** - Stable, sample efficient
2. **SAC (Soft Actor-Critic)** - For continuous action spaces
3. **DQN variants** - If discretizing action space

**Training Loop**:
```python
for epoch in range(num_epochs):
    for env_id in train_set:
        env = load_environment(env_id)
        trajectories = collect_trajectories(env, policy)
        loss = compute_loss(trajectories)
        update_policy(loss)
    
    if epoch % eval_interval == 0:
        eval_performance(val_set)
        save_checkpoint()
```

### 3.4 Reward Function Design
**File**: `rl_training/rewards.py`

**Components**:
```python
reward = (
    -1.0 * fire_spread_area        # Penalize fire growth
    -0.5 * property_damage          # Penalize building damage
    -0.1 * suppression_cost         # Penalize resource use
    +10.0 * containment_bonus       # Reward containment
    +5.0 * early_intervention       # Reward quick response
)
```

### 3.5 Simulation vs. Real Transitions
**Challenge**: Real fire data shows actual outcomes, not counterfactual "what if we intervened"

**Solutions**:
1. **Hybrid Approach**:
   - Train fire spread model separately
   - Use RL to learn suppression policy
   - Combine in environment

2. **Inverse RL**:
   - Learn reward from actual suppression data
   - Infer what firefighters optimize for

3. **Offline RL**:
   - Train from logged data only
   - Use conservative policy updates

---

## Phase 4: Evaluation & Deployment (FUTURE)
**Directory**: `evaluation/`

### 4.1 Offline Evaluation
**Script**: `evaluation/test_policy.py`

- Run trained policy on test set
- Metrics:
  - Fire containment rate
  - Time to containment
  - Total burned area
  - Resource efficiency

### 4.2 Visualization
**Script**: `evaluation/visualize_episodes.py`

- Animate fire spread and suppression actions
- Compare policy performance to baseline (no action, random, heuristic)
- Generate summary plots and statistics

### 4.3 Ablation Studies
- Vary reward function weights
- Test with different data modalities removed
- Analyze failure cases

### 4.4 Real-world Deployment (Long-term)
- API endpoint for real-time prediction
- Integration with fire management systems
- Continuous learning from new fire data

---

## Technical Requirements

### Compute Resources
- **Embedding Phase**: CPU-heavy (parallel processing of shapefiles)
  - Est. 32-64GB RAM for large rasters
  - Multi-core CPU for parallel tile processing

- **RL Training Phase**: GPU-heavy
  - Minimum: RTX 3080 or better
  - Recommended: A100 or multi-GPU setup
  - CUDA 12.x compatible

### Software Dependencies
- **Embedding**: See `embedding_src/requirements_embedding.txt`
  - rasterio, geopandas, pyproj, pandas, numpy
  
- **RL Training**: (To be created)
  - PyTorch / JAX / TensorFlow
  - Stable-Baselines3 / RLlib / CleanRL
  - gym / gymnasium

### Data Storage
- Raw data: ~50-100GB
- Embedded data: ~10-20GB (compressed)
- Tiled environments: ~20-50GB
- Model checkpoints: ~1-5GB
- Total: ~100-200GB recommended

---

## Current Status

| Phase | Status | Progress | Next Steps |
|-------|-------|----------|-----------|
| Data Embedding | Ready | 100% | Execute pipeline |
| Tiling | Design | 0% | Implement scripts (See tilling_src/PIPELINE.md) |
| RL Infrastructure | Planned | 0% | Design architecture |
| Training | Planned | 0% | Choose algorithm |
| Evaluation | Planned | 0% | Define metrics |

---

## Key Design Decisions

### 1. Why Tile-based Training?
- **Memory**: Full grid too large for GPU memory
- **Sample Efficiency**: More training samples from spatial diversity
- **Parallelization**: Train on multiple regions simultaneously
- **Generalization**: Test on unseen geographic regions

### 2. Why Episode-based Structure?
- **Real Fires**: Each fire is a natural episode with start/end
- **Credit Assignment**: Easier to attribute rewards to actions
- **Diversity**: Different fire types, seasons, locations

### 3. Spatial Resolution Trade-off
- **Higher Resolution** (e.g., 30m):
  - More detail, better for tactical decisions
  - Larger memory, slower training
  
- **Lower Resolution** (e.g., 90m-1km):
  - Faster training, smaller models
  - Better for strategic decisions
  
- **Current**: 90m (matching DEM resolution)

### 4. Temporal Resolution Trade-off
- **Hourly**: Captures rapid fire changes, more computational cost
- **6-hourly**: Balance between detail and efficiency
- **Daily**: Strategic planning, less tactical detail

---

## Future Extensions

1. **Multi-scale Hierarchy**:
   - Strategic model: Low-res, long-term planning
   - Tactical model: High-res, immediate actions
   - Hierarchical RL to combine

2. **Multi-agent**:
   - Multiple suppression teams
   - Coordination and communication

3. **Uncertainty Quantification**:
   - Probabilistic fire spread predictions
   - Risk-aware decision making

4. **Transfer Learning**:
   - Pre-train on simulation
   - Fine-tune on real data
   - Adapt to new regions

5. **Online Learning**:
   - Update model as new fires occur
   - Continual learning without forgetting

---

## References

### Fire Spread Models
- Rothermel (1972) - Classic fire spread equations
- FARSITE, FlamMap - Operational simulators
- ML-based: GNNs, CNNs for fire prediction

### RL for Disaster Management
- Wildfire suppression: Limited prior work
- Disaster response: RL for resource allocation
- Spatial decision making: RL on gridworld environments

### Relevant Papers
- "Deep RL for Wildfire Suppression" (if exists)
- "Spatiotemporal RL for Geographic Problems"
- "Multi-scale RL for Hierarchical Planning"

---

**Last Updated**: November 4, 2025  
**Next Review**: After embedding completion

