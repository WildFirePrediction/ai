# Phase 2: Environment Tiling - Master Plan

## Overview

**Purpose**: Transform embedded spatial data and temporal fire episodes into discrete, manageable RL training environments.

**Input**:
- Static spatial features (DEM, RSP, NDVI, LCM, FSM) on 658km × 684km grid
- 2,686 fire episodes with temporal weather data
- 115,799 fire detections spanning 2012-2025

**Output**:
- Thousands of small (256×256 pixel = ~102km × 102km) tile environments
- Each environment contains complete static features + temporal fire sequences
- Ready-to-use for A3C RL training

---

## Why Tiling?

### Problem: The Grid is Too Large

1. **Memory**: Full grid (1645 × 1710 × 3 features) = ~32 MB per timestep
   - With temporal weather: ~50 MB per timestep
   - A single episode: ~50 MB × 100 timesteps = 5 GB
   - 2,686 episodes: **13 TB** of data (!!)

2. **Compute**: RL agent must process entire grid
   - CNN on 1645×1710 is extremely slow
   - Policy network would be massive
   - Training time: months on single GPU

3. **Geographic Diversity**: Single episode often covers small area
   - 90% of grid is irrelevant to that fire
   - Wasting compute on empty cells

### Solution: Tile-Based Environments

1. **Small tiles** (256×256 = 102km × 102km)
   - Manageable memory: ~4 MB per timestep
   - Fast CNN inference: ~10ms on GPU
   - Covers typical fire extent (1-50km)

2. **Overlapping tiles** (50% overlap, stride=128)
   - Fire at tile edge appears in 4 tiles
   - Increases training samples by 4x
   - Better generalization (same fire, different contexts)

3. **Episode-tile assignment**
   - Each fire episode assigned to relevant tiles
   - Multiple tiles per episode (large fires)
   - Multiple episodes per tile (different times)

---

## Architecture Overview

```
Phase 1 (Complete):
  embedded_data/
  ├── state_vectors.npz            (Static: DEM, RSP, NDVI)
  ├── nasa_viirs_with_weather_reclustered.parquet  (Fire + Weather)
  └── episode_index_reclustered.parquet

Phase 2 (This Plan):
  tilling_data/
  ├── tile_index.parquet           (Tile metadata & episode assignments)
  ├── tiles/
  │   ├── tile_0000.npz            (Static features for tile 0)
  │   ├── tile_0001.npz
  │   └── ...
  ├── fire_sequences/
  │   ├── episode_0002_tile_0153.npz  (Temporal fire sequence)
  │   ├── episode_0002_tile_0154.npz
  │   └── ...
  └── environments/
      ├── env_manifest.json        (Environment catalog)
      ├── train_split.json         (70% train environments)
      ├── val_split.json           (15% validation)
      └── test_split.json          (15% test)

Phase 3 (Future):
  RL Training uses environments directly via PyTorch DataLoader
```

---

## Detailed Pipeline

### Step 1: Spatial Tiling

**Script**: `01_spatial_tiling.py`

**Objective**: Divide the full grid into overlapping tiles and identify which tiles are "interesting" (contain fire episodes).

**Process**:

1. **Define tile grid**:
   - Tile size: 256 × 256 pixels = 102.4km × 102.4km @ 400m resolution
   - Stride: 128 pixels (50% overlap)
   - Total tiles: `(1645 - 256) // 128 + 1` × `(1710 - 256) // 128 + 1` = ~11 × 12 = **~132 tiles**

2. **Tile coordinate system**:
   ```python
   tile_id = i * n_cols + j  # where i=row, j=col

   tile_bounds = {
       'x_start': x0 + j * stride * 400,      # EPSG:5179 meters
       'x_end':   x0 + (j * stride + 256) * 400,
       'y_start': y0 + i * stride * 400,
       'y_end':   y0 + (i * stride + 256) * 400,
       'i_start': i * stride,                  # Grid pixel indices
       'i_end':   i * stride + 256,
       'j_start': j * stride,
       'j_end':   j * stride + 256
   }
   ```

3. **Extract static features for each tile**:
   ```python
   # From state_vectors.npz
   dem = state_vectors['continuous_features'][0, i_start:i_end, j_start:j_end]
   rsp = state_vectors['continuous_features'][1, i_start:i_end, j_start:j_end]
   ndvi = state_vectors['continuous_features'][2, i_start:i_end, j_start:j_end]
   lcm = state_vectors['lcm_classes'][i_start:i_end, j_start:j_end]
   fsm = state_vectors['fsm_classes'][i_start:i_end, j_start:j_end]

   # Save as tile_XXXX.npz
   np.savez_compressed(f'tile_{tile_id:04d}.npz',
       dem=dem, rsp=rsp, ndvi=ndvi, lcm=lcm, fsm=fsm,
       bounds=tile_bounds, grid_coords=(i_start, i_end, j_start, j_end))
   ```

4. **Assign fire episodes to tiles**:
   - For each episode in `episode_index_reclustered.parquet`
   - Check spatial intersection with each tile using episode bounds (x_min, x_max, y_min, y_max)
   - Assign episode to tile if ANY fire detection falls within tile bounds

   ```python
   for episode_id, episode_info in episodes.iterrows():
       for tile_id, tile_bounds in tiles.iterrows():
           # Check if episode bounds intersect tile bounds
           if (episode_info['x_max'] >= tile_bounds['x_start'] and
               episode_info['x_min'] <= tile_bounds['x_end'] and
               episode_info['y_max'] >= tile_bounds['y_start'] and
               episode_info['y_min'] <= tile_bounds['y_end']):

               # Assign episode to this tile
               tile_episodes[tile_id].append(episode_id)
   ```

5. **Filter tiles**:
   - **Keep tiles with**: ≥1 fire episode
   - **Discard tiles with**:
     - No fire episodes (no training signal)
     - >80% nodata/ocean (insufficient valid data)
   - Expected: ~50-80 tiles (out of 132)

6. **Calculate tile statistics**:
   ```python
   tile_stats = {
       'num_episodes': len(tile_episodes[tile_id]),
       'num_detections': sum(ep_detections for ep in tile_episodes[tile_id]),
       'land_cover_diversity': shannon_entropy(lcm_tile),
       'forest_coverage_pct': (fsm_tile > 0).sum() / fsm_tile.size * 100,
       'mean_elevation': dem_tile[dem_tile > 0].mean(),
       'mean_slope': rsp_tile[rsp_tile > 0].mean(),
       'has_urban': (lcm_tile == URBAN_CLASS).any(),
       'fire_density': num_detections / 256**2  # detections per pixel
   }
   ```

**Output**: `tilling_data/tile_index.parquet`

| Column | Type | Description |
|--------|------|-------------|
| tile_id | int | Unique tile identifier (0-N) |
| i_row | int | Tile row index in grid |
| j_col | int | Tile column index in grid |
| x_start, x_end | float | EPSG:5179 X bounds (meters) |
| y_start, y_end | float | EPSG:5179 Y bounds (meters) |
| i_start, i_end | int | Grid pixel row indices |
| j_start, j_end | int | Grid pixel column indices |
| num_episodes | int | Number of fire episodes in tile |
| episode_ids | list[int] | Episode IDs assigned to tile |
| num_detections | int | Total fire detections in tile |
| land_cover_diversity | float | Shannon entropy of land cover |
| forest_coverage_pct | float | % forested area |
| mean_elevation | float | Mean DEM value |
| mean_slope | float | Mean RSP value |
| has_urban | bool | Contains urban areas |
| fire_density | float | Fire detections per pixel |

---

### Step 2: Temporal Segmentation

**Script**: `02_temporal_segmentation.py`

**Objective**: For each (episode, tile) pair, create temporal sequences of fire progression at regular intervals.

**Process**:

1. **For each episode assigned to each tile**:
   - Get all fire detections in that episode within tile bounds
   - Sort by timestamp
   - Determine episode start/end times

2. **Create timesteps at fixed intervals**:
   - **Timestep interval**: 6 hours (adjustable)
   - **Why 6 hours?**:
     - Fire behavior changes on ~6h cycle (day/night)
     - VIIRS satellite revisit: ~6-12 hours
     - Operational decision-making timeframe
     - Not too sparse (missing dynamics), not too dense (redundant)

   ```python
   start_time = episode_start
   end_time = episode_end
   timesteps = pd.date_range(start_time, end_time, freq='6H')

   # For each timestep, create a "snapshot" of the fire state
   for t in timesteps:
       # Window: [t - 3h, t + 3h] to account for satellite revisit variance
       window_start = t - pd.Timedelta(hours=3)
       window_end = t + pd.Timedelta(hours=3)

       detections_in_window = episode_detections[
           (episode_detections['datetime'] >= window_start) &
           (episode_detections['datetime'] <= window_end)
       ]
   ```

3. **Create fire state rasters for each timestep**:

   **Fire Mask** (256 × 256, binary):
   - 1 = active fire detected
   - 0 = no fire
   ```python
   fire_mask = np.zeros((256, 256), dtype=np.uint8)
   for detection in detections_in_window:
       # Convert world coords to tile pixel coords
       i = int((detection['y'] - tile_y_start) / 400)
       j = int((detection['x'] - tile_x_start) / 400)
       if 0 <= i < 256 and 0 <= j < 256:
           fire_mask[i, j] = 1
   ```

   **Fire Intensity** (256 × 256, float):
   - Normalized fire intensity (from FRP + Brightness)
   ```python
   fire_intensity = np.zeros((256, 256), dtype=np.float32)
   for detection in detections_in_window:
       i, j = world_to_tile_coords(detection['x'], detection['y'])
       fire_intensity[i, j] = detection['i_norm']  # z-score normalized
   ```

   **Fire Temperature** (256 × 256, float):
   ```python
   fire_temp = np.zeros((256, 256), dtype=np.float32)
   for detection in detections_in_window:
       i, j = world_to_tile_coords(detection['x'], detection['y'])
       fire_temp[i, j] = detection['te_norm']
   ```

   **Fire Age** (256 × 256, float):
   - Hours since first detection in episode
   ```python
   fire_age = np.zeros((256, 256), dtype=np.float32)
   for detection in detections_in_window:
       i, j = world_to_tile_coords(detection['x'], detection['y'])
       age_hours = (detection['datetime'] - episode_start).total_seconds() / 3600
       fire_age[i, j] = age_hours
   ```

4. **Aggregate weather for each timestep**:
   - Weather is per-detection (not gridded)
   - Aggregate detections within timestep window:

   ```python
   if len(detections_in_window) > 0:
       weather_state = {
           'wind_speed': detections_in_window['w_norm'].mean(),
           'wind_dir_x': detections_in_window['d_x_norm'].mean(),
           'wind_dir_y': detections_in_window['d_y_norm'].mean(),
           'humidity': detections_in_window['rh_norm'].mean(),
           'precipitation': detections_in_window['r_norm'].mean()
       }
   else:
       # No detections in this timestep (fire dormant or extinguished)
       weather_state = {
           'wind_speed': 0.0,
           'wind_dir_x': 0.0,
           'wind_dir_y': 0.0,
           'humidity': 0.0,
           'precipitation': 0.0
       }
   ```

5. **Handle edge cases**:
   - **No detections in timestep**: Fire dormant, keep previous state
   - **Episode end**: Last timestep = fire contained/extinguished
   - **Sparse detections**: Use nearest neighbor interpolation for weather

6. **Save temporal sequence**:
   ```python
   sequence = {
       'episode_id': episode_id,
       'tile_id': tile_id,
       'timesteps': timesteps,  # List of datetime
       'fire_masks': fire_masks,  # (T, 256, 256) uint8
       'fire_intensity': fire_intensity,  # (T, 256, 256) float32
       'fire_temp': fire_temp,  # (T, 256, 256) float32
       'fire_age': fire_age,  # (T, 256, 256) float32
       'weather': weather_states,  # (T, 5) float32 [w, dx, dy, rh, r]
       'metadata': {
           'start_time': episode_start,
           'end_time': episode_end,
           'num_timesteps': len(timesteps),
           'num_detections': len(episode_detections),
           'max_intensity': fire_intensity.max()
       }
   }

   np.savez_compressed(
       f'fire_sequences/episode_{episode_id:04d}_tile_{tile_id:04d}.npz',
       **sequence
   )
   ```

**Output**: `tilling_data/fire_sequences/episode_XXXX_tile_YYYY.npz` (one file per episode-tile pair)

**Expected files**: ~2,686 episodes × ~2 tiles/episode = **~5,000-8,000 sequence files**

---

### Step 3: Environment Assembly

**Script**: `03_environment_assembly.py`

**Objective**: Combine static tile features + temporal fire sequences into complete RL environments.

**Process**:

1. **For each fire sequence file** (`episode_X_tile_Y.npz`):
   - Load static tile features from `tiles/tile_Y.npz`
   - Load temporal sequence from `fire_sequences/episode_X_tile_Y.npz`
   - Combine into unified environment

2. **Environment state structure**:
   ```python
   class WildfireEnvironment:
       def __init__(self, episode_id, tile_id):
           # Static features (constant across timesteps)
           self.static = {
               'dem': (256, 256),        # Elevation
               'rsp': (256, 256),        # Slope
               'ndvi': (256, 256),       # Vegetation
               'lcm': (256, 256),        # Land cover (categorical)
               'fsm': (256, 256)         # Forest type (categorical)
           }

           # Dynamic features (change per timestep)
           self.temporal = {
               'fire_mask': (T, 256, 256),       # Active fire locations
               'fire_intensity': (T, 256, 256),  # Fire intensity
               'fire_temp': (T, 256, 256),       # Fire temperature
               'fire_age': (T, 256, 256),        # Hours since ignition
               'weather': (T, 5)                 # Weather state vector
           }

           # Metadata
           self.metadata = {
               'episode_id': int,
               'tile_id': int,
               'num_timesteps': int,
               'start_time': datetime,
               'end_time': datetime,
               'tile_bounds': dict
           }
   ```

3. **Create observation space** (what RL agent sees):
   ```python
   def get_observation(self, timestep_idx):
       """
       Returns observation at timestep t
       Shape: (C, 256, 256) where C = num_channels
       """

       # Static features (5 channels)
       static_obs = np.stack([
           self.static['dem'],
           self.static['rsp'],
           self.static['ndvi'],
           self.static['lcm'],  # Will be embedded in NN
           self.static['fsm']   # Will be embedded in NN
       ], axis=0)  # Shape: (5, 256, 256)

       # Dynamic fire features (4 channels)
       fire_obs = np.stack([
           self.temporal['fire_mask'][timestep_idx],
           self.temporal['fire_intensity'][timestep_idx],
           self.temporal['fire_temp'][timestep_idx],
           self.temporal['fire_age'][timestep_idx]
       ], axis=0)  # Shape: (4, 256, 256)

       # Weather (broadcast to spatial dims or use as global context)
       # Option A: Broadcast to grid
       weather = self.temporal['weather'][timestep_idx]  # (5,)
       weather_grid = np.tile(weather[:, None, None], (1, 256, 256))  # (5, 256, 256)

       # Option B: Keep as vector (NN will handle)
       # weather_vector = weather  # (5,) - global context

       # Combine all
       observation = np.concatenate([
           static_obs,   # (5, 256, 256)
           fire_obs,     # (4, 256, 256)
           weather_grid  # (5, 256, 256)
       ], axis=0)  # Total: (14, 256, 256)

       return observation
   ```

4. **Define action space**:
   ```python
   # Option A: Discrete spatial action (select cell for suppression)
   action_space = gym.spaces.Discrete(256 * 256)  # 65,536 actions

   # Option B: Multi-discrete (select cell + action type)
   action_space = gym.spaces.MultiDiscrete([256, 256, 4])  # row, col, action_type
   # action_types: 0=none, 1=water_drop, 2=fire_line, 3=controlled_burn

   # Option C: Continuous (normalized coordinates + intensity)
   action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))  # x, y, intensity
   ```

5. **Define reward function**:
   ```python
   def compute_reward(self, state_t, action, state_t1):
       """
       Reward = -(fire_growth) - (property_damage) - (suppression_cost) + (containment_bonus)
       """

       # Fire spread penalty
       fire_area_t = (state_t['fire_mask'] > 0).sum()
       fire_area_t1 = (state_t1['fire_mask'] > 0).sum()
       fire_growth = fire_area_t1 - fire_area_t

       # Property damage (fire in urban/residential areas)
       urban_mask = (state_t1['lcm'] == URBAN_CLASS)
       urban_damage = (state_t1['fire_mask'] * urban_mask).sum()

       # Suppression cost (resources used)
       suppression_cost = action_intensity * cost_per_unit

       # Containment bonus (fire perimeter not expanding)
       contained = (fire_area_t1 == fire_area_t) and (fire_area_t1 > 0)
       containment_bonus = 100 if contained else 0

       # Extinction bonus (fire completely out)
       extinct = (fire_area_t1 == 0) and (fire_area_t > 0)
       extinction_bonus = 500 if extinct else 0

       # Early intervention bonus
       early_intervention = max(0, (T_max - timestep_idx) / T_max) * 50

       reward = (
           -1.0 * fire_growth +           # Penalize spread
           -10.0 * urban_damage +         # Heavily penalize urban damage
           -0.1 * suppression_cost +      # Small cost for resources
           containment_bonus +            # Reward containment
           extinction_bonus +             # Large reward for extinction
           early_intervention             # Reward quick action
       )

       return reward
   ```

6. **Save environment**:
   ```python
   environment = {
       'episode_id': episode_id,
       'tile_id': tile_id,
       'static_features': static_features,
       'temporal_sequence': temporal_sequence,
       'metadata': metadata,
       'observation_shape': (14, 256, 256),
       'action_space_type': 'discrete',
       'num_timesteps': T
   }

   # Save as pickle (for complex Python objects) or NPZ (for arrays)
   import pickle
   with open(f'environments/env_{episode_id:04d}_{tile_id:04d}.pkl', 'wb') as f:
       pickle.dump(environment, f)
   ```

**Output**: `tilling_data/environments/env_XXXX_YYYY.pkl` (one file per environment)

---

### Step 4: Dataset Splitting

**Script**: `04_dataset_split.py`

**Objective**: Split environments into train/validation/test sets with stratification to ensure balanced distribution.

**Process**:

1. **Load environment manifest**:
   ```python
   env_files = list(Path('environments/').glob('env_*.pkl'))

   # Extract metadata
   env_metadata = []
   for env_file in env_files:
       with open(env_file, 'rb') as f:
           env = pickle.load(f)

       env_metadata.append({
           'env_id': env_file.stem,
           'episode_id': env['episode_id'],
           'tile_id': env['tile_id'],
           'num_timesteps': env['num_timesteps'],
           'fire_size': env['metadata']['num_detections'],
           'season': get_season(env['metadata']['start_time']),
           'region': get_region(env['metadata']['tile_bounds']),
           'duration_days': env['metadata']['duration_days']
       })

   df_envs = pd.DataFrame(env_metadata)
   ```

2. **Stratification criteria**:
   - **Fire size**: Small (<10 detections), Medium (10-100), Large (>100)
   - **Season**: Spring (Mar-May), Summer (Jun-Aug), Fall (Sep-Nov), Winter (Dec-Feb)
   - **Region**: North, Central, South (divide Y coordinate into thirds)
   - **Duration**: Short (<1 day), Medium (1-7 days), Long (>7 days)

   ```python
   # Categorize
   df_envs['fire_size_cat'] = pd.cut(df_envs['fire_size'],
                                     bins=[0, 10, 100, np.inf],
                                     labels=['small', 'medium', 'large'])

   df_envs['duration_cat'] = pd.cut(df_envs['duration_days'],
                                    bins=[0, 1, 7, np.inf],
                                    labels=['short', 'medium', 'long'])

   # Create stratification key
   df_envs['strat_key'] = (df_envs['fire_size_cat'].astype(str) + '_' +
                           df_envs['season'].astype(str) + '_' +
                           df_envs['region'].astype(str))
   ```

3. **Perform stratified split**:
   ```python
   from sklearn.model_selection import StratifiedShuffleSplit

   splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

   # First split: 70% train, 30% temp
   train_idx, temp_idx = next(splitter.split(df_envs, df_envs['strat_key']))

   # Second split: 15% val, 15% test (from 30% temp)
   df_temp = df_envs.iloc[temp_idx]
   splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
   val_idx_temp, test_idx_temp = next(splitter2.split(df_temp, df_temp['strat_key']))

   val_idx = temp_idx[val_idx_temp]
   test_idx = temp_idx[test_idx_temp]

   # Create splits
   train_envs = df_envs.iloc[train_idx]['env_id'].tolist()
   val_envs = df_envs.iloc[val_idx]['env_id'].tolist()
   test_envs = df_envs.iloc[test_idx]['env_id'].tolist()
   ```

4. **Ensure no spatial overlap** (critical!):
   - Tiles with 50% overlap → same fire may appear in adjacent tiles
   - Must ensure train/val/test don't share overlapping tiles

   ```python
   def get_overlapping_tiles(tile_id, tile_index):
       """Find tiles that spatially overlap with given tile"""
       tile = tile_index[tile_index['tile_id'] == tile_id].iloc[0]

       overlapping = tile_index[
           (tile_index['x_start'] < tile['x_end']) &
           (tile_index['x_end'] > tile['x_start']) &
           (tile_index['y_start'] < tile['y_end']) &
           (tile_index['y_end'] > tile['y_start'])
       ]

       return overlapping['tile_id'].tolist()

   # Remove overlapping tiles from validation/test if in train
   train_tiles = set(df_envs.iloc[train_idx]['tile_id'])

   for tile in train_tiles:
       overlapping = get_overlapping_tiles(tile, tile_index_df)

       # Remove from val/test
       val_envs = [e for e in val_envs if df_envs[df_envs['env_id']==e]['tile_id'].iloc[0] not in overlapping]
       test_envs = [e for e in test_envs if df_envs[df_envs['env_id']==e]['tile_id'].iloc[0] not in overlapping]
   ```

5. **Validate split quality**:
   ```python
   def print_split_stats(split_name, env_ids, df_envs):
       split_df = df_envs[df_envs['env_id'].isin(env_ids)]

       print(f"\n{split_name} Split:")
       print(f"  Total environments: {len(env_ids)}")
       print(f"  Fire size distribution:")
       print(split_df['fire_size_cat'].value_counts(normalize=True))
       print(f"  Season distribution:")
       print(split_df['season'].value_counts(normalize=True))
       print(f"  Region distribution:")
       print(split_df['region'].value_counts(normalize=True))

   print_split_stats("Train", train_envs, df_envs)
   print_split_stats("Validation", val_envs, df_envs)
   print_split_stats("Test", test_envs, df_envs)
   ```

6. **Save splits**:
   ```python
   splits = {
       'train': train_envs,
       'val': val_envs,
       'test': test_envs,
       'split_date': datetime.now().isoformat(),
       'random_seed': 42,
       'stratification': 'fire_size × season × region',
       'num_train': len(train_envs),
       'num_val': len(val_envs),
       'num_test': len(test_envs)
   }

   with open('environments/train_split.json', 'w') as f:
       json.dump({'env_ids': train_envs, 'metadata': splits}, f, indent=2)

   with open('environments/val_split.json', 'w') as f:
       json.dump({'env_ids': val_envs, 'metadata': splits}, f, indent=2)

   with open('environments/test_split.json', 'w') as f:
       json.dump({'env_ids': test_envs, 'metadata': splits}, f, indent=2)

   # Also save manifest
   with open('environments/env_manifest.json', 'w') as f:
       json.dump({
           'total_environments': len(env_files),
           'train': len(train_envs),
           'val': len(val_envs),
           'test': len(test_envs),
           'environment_metadata': env_metadata
       }, f, indent=2)
   ```

**Output**:
- `environments/train_split.json` (~70% of environments)
- `environments/val_split.json` (~15%)
- `environments/test_split.json` (~15%)
- `environments/env_manifest.json` (complete catalog)

---

### Step 5: Environment Validation

**Script**: `05_environment_validation.py`

**Objective**: Quality checks and visualizations to ensure environments are correct before training.

**Process**:

1. **Load sample environments**:
   ```python
   sample_envs = random.sample(train_envs, 10)

   for env_id in sample_envs:
       with open(f'environments/{env_id}.pkl', 'rb') as f:
           env = pickle.load(f)

       # Validate structure
       assert 'static_features' in env
       assert 'temporal_sequence' in env
       assert env['observation_shape'] == (14, 256, 256)
   ```

2. **Check data quality**:
   - No NaN/Inf values in observations
   - Fire masks are binary (0 or 1)
   - Weather values in reasonable ranges
   - Temporal continuity (no sudden jumps)
   - Spatial coherence (fire doesn't teleport)

3. **Visualize sample episodes**:
   ```python
   def visualize_environment(env, output_path):
       fig, axes = plt.subplots(4, 4, figsize=(16, 16))

       # Row 1: Static features
       axes[0, 0].imshow(env['static']['dem'], cmap='terrain')
       axes[0, 0].set_title('Elevation (DEM)')

       axes[0, 1].imshow(env['static']['rsp'], cmap='viridis')
       axes[0, 1].set_title('Slope (RSP)')

       axes[0, 2].imshow(env['static']['ndvi'], cmap='Greens')
       axes[0, 2].set_title('Vegetation (NDVI)')

       axes[0, 3].imshow(env['static']['lcm'], cmap='tab10')
       axes[0, 3].set_title('Land Cover')

       # Row 2-4: Fire progression (select 12 timesteps evenly)
       timesteps = np.linspace(0, env['num_timesteps']-1, 12, dtype=int)

       for idx, t in enumerate(timesteps):
           ax = axes[idx // 4 + 1, idx % 4]

           # Composite: base map + fire overlay
           base = env['static']['dem']
           fire = env['temporal']['fire_mask'][t]

           ax.imshow(base, cmap='gray', alpha=0.5)
           ax.imshow(fire, cmap='hot', alpha=0.7, vmin=0, vmax=1)
           ax.set_title(f't = {t*6}h')

       plt.tight_layout()
       plt.savefig(output_path, dpi=150)
   ```

4. **Generate validation report**:
   ```python
   report = {
       'num_environments': len(env_files),
       'observation_shape': (14, 256, 256),
       'action_space': 'discrete',
       'temporal_resolution': '6 hours',
       'spatial_resolution': '400m',
       'tile_size': '256x256 pixels = 102.4km x 102.4km',
       'checks': {
           'no_nan_values': True,
           'fire_mask_binary': True,
           'weather_in_range': True,
           'temporal_continuity': True,
           'spatial_coherence': True
       },
       'issues': []
   }
   ```

**Output**:
- `validation/environment_samples.png` (visualization)
- `validation/validation_report.json`

---

## Connection to RL Training (Phase 3)

### PyTorch DataLoader Integration

```python
# rl_training/data_loader.py

class WildfireDataset(torch.utils.data.Dataset):
    def __init__(self, split='train'):
        # Load environment IDs from split JSON
        with open(f'tilling_data/environments/{split}_split.json') as f:
            self.env_ids = json.load(f)['env_ids']

    def __len__(self):
        return len(self.env_ids)

    def __getitem__(self, idx):
        env_id = self.env_ids[idx]

        # Load environment
        with open(f'tilling_data/environments/{env_id}.pkl', 'rb') as f:
            env = pickle.load(f)

        # Return as tensors
        return {
            'static': torch.FloatTensor(env['static_features']),
            'temporal': torch.FloatTensor(env['temporal_sequence']),
            'metadata': env['metadata']
        }

# Usage
train_dataset = WildfireDataset(split='train')
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

for batch in train_loader:
    # Train A3C agent
    ...
```

### Gym Environment Wrapper

```python
# rl_training/wildfire_env.py

class WildfireEnv(gym.Env):
    def __init__(self, env_data):
        self.static = env_data['static_features']
        self.temporal = env_data['temporal_sequence']
        self.num_timesteps = env_data['num_timesteps']
        self.current_timestep = 0

        # Define spaces
        self.observation_space = gym.spaces.Box(
            low=-5, high=5, shape=(14, 256, 256), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(256 * 256)

    def reset(self):
        self.current_timestep = 0
        return self._get_observation()

    def step(self, action):
        # Apply suppression action
        # Update fire state
        # Calculate reward

        self.current_timestep += 1
        obs = self._get_observation()
        reward = self._calculate_reward(action)
        done = (self.current_timestep >= self.num_timesteps)

        return obs, reward, done, {}

    def _get_observation(self):
        # Combine static + dynamic features at current timestep
        static_obs = np.stack([
            self.static['dem'],
            self.static['rsp'],
            self.static['ndvi'],
            self.static['lcm'],
            self.static['fsm']
        ], axis=0)

        fire_obs = np.stack([
            self.temporal['fire_mask'][self.current_timestep],
            self.temporal['fire_intensity'][self.current_timestep],
            self.temporal['fire_temp'][self.current_timestep],
            self.temporal['fire_age'][self.current_timestep]
        ], axis=0)

        weather = self.temporal['weather'][self.current_timestep]
        weather_grid = np.tile(weather[:, None, None], (1, 256, 256))

        return np.concatenate([static_obs, fire_obs, weather_grid], axis=0)
```

---

## Technical Specifications

### Memory Requirements

| Component | Size per Item | Count | Total |
|-----------|---------------|-------|-------|
| Static tile features | 4 MB | 80 tiles | 320 MB |
| Fire sequence (100 timesteps) | 40 MB | 5,000 seqs | 200 GB |
| Environment pickle | 50 MB | 5,000 envs | 250 GB |
| **Total Storage** | | | **~450 GB** |

### Compute Requirements

- **Tiling Phase**: ~2-4 hours on single CPU core
- **Temporal Segmentation**: ~6-8 hours (parallelizable)
- **Environment Assembly**: ~2-4 hours
- **Total Pipeline Time**: ~12-16 hours

### Parallelization Strategy

```python
# Use multiprocessing for temporal segmentation (slowest step)
from multiprocessing import Pool

def process_episode_tile_pair(episode_id, tile_id):
    # Load data
    # Create temporal sequence
    # Save result
    pass

# Parallel execution
with Pool(processes=16) as pool:
    pool.starmap(process_episode_tile_pair, episode_tile_pairs)
```

---

## Expected Outputs Summary

After completing all 5 scripts:

```
tilling_data/
├── tile_index.parquet              (~1 MB)     80 tiles
├── tiles/
│   ├── tile_0000.npz               (4 MB each) 80 files = 320 MB
│   └── ...
├── fire_sequences/
│   ├── episode_0002_tile_0153.npz  (40 MB each) 5,000 files = 200 GB
│   └── ...
├── environments/
│   ├── env_0002_0153.pkl           (50 MB each) 5,000 files = 250 GB
│   ├── env_manifest.json
│   ├── train_split.json            (3,500 train envs)
│   ├── val_split.json              (750 val envs)
│   └── test_split.json             (750 test envs)
└── validation/
    ├── environment_samples.png
    └── validation_report.json
```

**Total**: ~450 GB of processed RL training data

---

## Next Steps After Tiling

1. **Implement Gym Environment** (`rl_training/wildfire_env.py`)
2. **Design Neural Network** (`rl_training/models.py`)
   - CNN encoder for spatial features
   - Embedding layers for LCM/FSM
   - Policy head + Value head for A3C
3. **Implement A3C Algorithm** (`rl_training/train.py`)
4. **Train on cluster/cloud** (requires GPU, days-weeks of training)
5. **Evaluate on test set**
6. **Deploy for inference**

---

## Critical Design Decisions

### 1. Tile Size: 256×256 pixels

**Rationale**:
- Typical fire extent: 0.5-50km → 256×400m = 102km covers most fires
- CNN receptive field: 256px allows ~20-layer CNN to see entire tile
- Memory: 4 MB per tile is manageable on GPU
- Batch size: 8-16 tiles fit in 12GB GPU memory

**Alternative considered**: 512×512 (too large, 16 MB/tile, slow CNN)

### 2. Temporal Resolution: 6 hours

**Rationale**:
- Satellite revisit: VIIRS ~6-12 hours
- Fire behavior cycle: day/night changes
- Operational decisions: 6h response window realistic
- Not too sparse (miss rapid changes), not too dense (redundant)

**Alternative considered**: 1 hour (too sparse data), 24 hours (too coarse)

### 3. Overlap: 50% (stride=128)

**Rationale**:
- Fire at edge appears in 4 tiles → 4x training samples
- Better generalization (same fire, different contexts)
- No info loss at tile boundaries

**Alternative considered**: No overlap (fewer samples, boundary artifacts)

### 4. Weather Aggregation: Mean within 6h window

**Rationale**:
- Weather changes slowly (~hourly)
- Detections within 6h window have similar weather
- Avoids gaps when no detection at exact timestep

**Alternative considered**: Interpolation (complex, may introduce artifacts)

---

## Potential Issues & Solutions

### Issue 1: Large episodes span multiple tiles

**Problem**: Episode with 100km extent touches 4-9 tiles

**Solution**: Assign episode to ALL overlapping tiles. Agent learns to manage fires from different viewpoints.

### Issue 2: Sparse fire detections

**Problem**: Some 6h windows have zero detections

**Solution**:
- Mark as "dormant" state (fire_mask=0, keep last weather)
- Alternative: Skip sparse timesteps (only use active periods)

### Issue 3: Unbalanced classes

**Problem**: Most timesteps have no fire (class imbalance)

**Solution**:
- Oversample episodes with active fires
- Use weighted loss function in RL training
- Curriculum learning: start with active fires, gradually add dormant states

### Issue 4: Storage (450 GB)

**Problem**: Large disk space requirement

**Solution**:
- Compress with `np.savez_compressed` (already done)
- Use float16 instead of float32 (2x reduction)
- Stream from disk during training (slower but saves RAM)
- Use cloud storage (S3, GCS) with caching

### Issue 5: Long processing time (12-16 hours)

**Problem**: Slow pipeline execution

**Solution**:
- Parallelize temporal segmentation (16 cores → 4x speedup)
- Use faster storage (SSD vs HDD)
- Process episodes in batches
- Cache intermediate results

---

## Validation Checklist

Before proceeding to Phase 3 (Training):

- [ ] All tiles have valid static features (no NaN)
- [ ] Fire sequences are temporally continuous
- [ ] Weather values in realistic ranges
- [ ] Train/val/test splits are balanced
- [ ] No spatial overlap between splits
- [ ] Visualizations show fire progression correctly
- [ ] Environment shape matches NN input spec
- [ ] Reward function implemented correctly
- [ ] Gym environment runs without errors
- [ ] DataLoader can load all environments

---

## Summary

**Input**: Embedded spatial data (1645×1710 grid) + 2,686 fire episodes
**Output**: ~5,000 RL training environments ready for A3C
**Time**: ~12-16 hours processing
**Storage**: ~450 GB
**Quality**: Validated, stratified, ready for GPU training

Once this phase is complete, we proceed to Phase 3 (RL Training) with confidence that the data is clean, well-structured, and suitable for learning effective wildfire management policies.
