# Tilling Pipeline for RL Environment Setup

## Overview
This pipeline converts embedded spatial-temporal wildfire data into discrete RL training environments. Each environment represents a spatial region (tile) with one or more fire episodes.

**Prerequisite**: Complete embedding pipeline (`embedding_src/`) must be executed first.

---

## Input Data Structure

### From Embedding Phase
Located in `../embedded_data/`:

1. **nasa_viirs_embedded.parquet**
   - Columns: `x`, `y`, `te`, `i`, `tm`, `episode_id`, `datetime`, normalized versions
   - Format: One row per fire detection
   - ~10K-100K rows

2. **dem_rsp_embedded.tif**
   - 2-band GeoTIFF (EPSG:5179)
   - Band 1: Normalized elevation
   - Band 2: Normalized relative slope position
   - Resolution: 90m

3. **lcm_embedded.tif**
   - 1-band GeoTIFF with class indices
   - Classes: Land cover types (see `lcm_class_mapping.json`)

4. **fsm_embedded.tif**
   - 1-band GeoTIFF with forest type indices
   - Classes: Forest stand types (see `fsm_class_mapping.json`)

5. **ndvi_embedded.tif**
   - Multi-band GeoTIFF with temporal NDVI values
   - Each band: Different time period

6. **kma_weather_embedded.tif**
   - 5-band GeoTIFF
   - Bands: wind_speed, wind_x, wind_y, humidity, precipitation

7. **metadata.json**
   - Grid dimensions, CRS, transform matrix
   - Feature descriptions

---

## Pipeline Steps

### Step 1: Spatial Tiling
**Script**: `01_spatial_tiling.py`

**Purpose**: Divide the full Korean grid into manageable spatial tiles

#### Configuration
```python
TILE_SIZE = 256          # Pixels per tile (256x256 = ~23km x 23km at 90m)
STRIDE = 128             # Overlap between tiles (50%)
MIN_FIRE_EPISODES = 1    # Minimum fires required per tile
MIN_VALID_PIXELS = 0.8   # Minimum fraction of non-null pixels
```

#### Process

1. **Load Grid Metadata**
   ```python
   with open('../embedded_data/metadata.json') as f:
       metadata = json.load(f)
   
   grid_height = metadata['height']
   grid_width = metadata['width']
   transform = metadata['transform']  # For pixel→coordinate conversion
   ```

2. **Generate Tile Grid**
   ```python
   tiles = []
   for y_start in range(0, grid_height - TILE_SIZE + 1, STRIDE):
       for x_start in range(0, grid_width - TILE_SIZE + 1, STRIDE):
           y_end = y_start + TILE_SIZE
           x_end = x_start + TILE_SIZE
           
           tile = {
               'tile_id': f"tile_{y_start}_{x_start}",
               'y_start': y_start, 'y_end': y_end,
               'x_start': x_start, 'x_end': x_end
           }
           tiles.append(tile)
   ```

3. **Convert to Geographic Coordinates**
   ```python
   for tile in tiles:
       # Top-left corner
       lon_min, lat_max = transform * (tile['x_start'], tile['y_start'])
       # Bottom-right corner
       lon_max, lat_min = transform * (tile['x_end'], tile['y_end'])
       
       tile.update({
           'lon_min': lon_min, 'lon_max': lon_max,
           'lat_min': lat_min, 'lat_max': lat_max
       })
   ```

4. **Load Fire Episode Locations**
   ```python
   df_fire = pd.read_parquet('../embedded_data/nasa_viirs_embedded.parquet')
   
   # Get episode bounding boxes
   episodes = df_fire.groupby('episode_id').agg({
       'x': ['min', 'max'],
       'y': ['min', 'max'],
       'datetime': ['min', 'max'],
       'i': 'mean'
   }).reset_index()
   ```

5. **Assign Episodes to Tiles**
   ```python
   for tile in tiles:
       # Find episodes that intersect this tile
       # (using spatial bounding box overlap)
       tile_episodes = []
       
       for _, ep in episodes.iterrows():
           if (ep['x_min'] <= tile['x_end'] and 
               ep['x_max'] >= tile['x_start'] and
               ep['y_min'] <= tile['y_end'] and 
               ep['y_max'] >= tile['y_start']):
               
               tile_episodes.append(ep['episode_id'])
       
       tile['fire_episode_ids'] = tile_episodes
       tile['num_fire_episodes'] = len(tile_episodes)
   ```

6. **Filter Valid Tiles**
   ```python
   # Load one raster to check valid pixels
   with rasterio.open('../embedded_data/dem_rsp_embedded.tif') as src:
       dem = src.read(1)
   
   valid_tiles = []
   for tile in tiles:
       if tile['num_fire_episodes'] < MIN_FIRE_EPISODES:
           continue
       
       # Extract tile region
       tile_dem = dem[
           tile['y_start']:tile['y_end'],
           tile['x_start']:tile['x_end']
       ]
       
       # Check valid pixel ratio
       valid_ratio = np.isfinite(tile_dem).mean()
       if valid_ratio >= MIN_VALID_PIXELS:
           tile['valid_pixel_ratio'] = valid_ratio
           valid_tiles.append(tile)
   ```

7. **Compute Tile Statistics**
   ```python
   for tile in valid_tiles:
       tile_region = extract_tile_region(tile)  # Helper function
       
       # Land cover diversity (Shannon entropy)
       lcm = tile_region['lcm']
       unique, counts = np.unique(lcm, return_counts=True)
       probs = counts / counts.sum()
       tile['land_cover_diversity'] = -np.sum(probs * np.log(probs))
       
       # Forest coverage
       forest_classes = [1, 2, 3]  # From lcm_class_mapping
       tile['forest_coverage'] = np.isin(lcm, forest_classes).mean()
       
       # Average elevation
       dem = tile_region['dem']
       tile['avg_elevation'] = np.nanmean(dem)
   ```

#### Output
**File**: `../tilled_data/tile_index.parquet`

Columns:
- `tile_id`: str - Unique identifier
- `x_start`, `x_end`, `y_start`, `y_end`: int - Grid pixel coordinates
- `lon_min`, `lon_max`, `lat_min`, `lat_max`: float - Geographic bounds
- `num_fire_episodes`: int - Count of fires in tile
- `fire_episode_ids`: list - Episode IDs intersecting tile
- `valid_pixel_ratio`: float - Fraction of non-null pixels
- `land_cover_diversity`: float - Shannon entropy
- `forest_coverage`: float - Percentage [0, 1]
- `avg_elevation`: float - Mean elevation (m)

**Summary**: `../tilled_data/tile_summary.txt`
```
Total tiles generated: 1234
Tiles with fires: 567
Total fire episodes: 890
Mean episodes per tile: 1.57
Tiles filtered out (insufficient data): 667
```

---

### Step 2: Fire Cluster Validation
**Script**: `02_fire_cluster_validation.py`

**Purpose**: Validate and refine fire episode clusters, ensure data quality

#### Process

1. **Load Fire Data and Tiles**
   ```python
   df_fire = pd.read_parquet('../embedded_data/nasa_viirs_embedded.parquet')
   tiles_df = pd.read_parquet('../tilled_data/tile_index.parquet')
   ```

2. **Episode Quality Checks**
   ```python
   for episode_id in df_fire['episode_id'].unique():
       ep_data = df_fire[df_fire['episode_id'] == episode_id]
       
       # Check 1: Minimum detections
       if len(ep_data) < 3:
           flag_episode(episode_id, 'too_few_detections')
       
       # Check 2: Spatial coherence (max distance between points)
       coords = ep_data[['x', 'y']].values
       pairwise_dist = scipy.spatial.distance.pdist(coords)
       if pairwise_dist.max() > 50000:  # 50km
           flag_episode(episode_id, 'spatially_dispersed')
       
       # Check 3: Temporal coherence (max time gap)
       times = ep_data['datetime'].sort_values()
       time_gaps = times.diff().dt.total_seconds() / 3600
       if time_gaps.max() > 72:  # 72 hours
           flag_episode(episode_id, 'temporal_gap')
       
       # Check 4: Progression makes sense (not jumping around)
       # ... additional logic ...
   ```

3. **Re-cluster if Needed**
   ```python
   # If too many episodes flagged, consider re-running DBSCAN
   # with different parameters on the specific regions
   ```

4. **Update Tile Index**
   ```python
   # Remove flagged episodes from tile assignments
   # Update episode counts
   ```

#### Output
- Updated `../tilled_data/tile_index.parquet`
- `../tilled_data/episode_flags.csv` - Quality issues per episode
- `../tilled_data/cluster_validation_report.txt`

---

### Step 3: Temporal Segmentation
**Script**: `03_temporal_segmentation.py`

**Purpose**: Create temporal sequences for each fire episode

#### Configuration
```python
TIME_BEFORE_FIRE = 24    # Hours before first detection
TIME_AFTER_FIRE = 168    # Hours after last detection (7 days)
TIMESTEP_HOURS = 6       # Interval between snapshots
```

#### Process

1. **Load Data**
   ```python
   df_fire = pd.read_parquet('../embedded_data/nasa_viirs_embedded.parquet')
   tiles_df = pd.read_parquet('../tilled_data/tile_index.parquet')
   ```

2. **For Each Tile-Episode Pair**
   ```python
   sequences_dir = Path('../tilled_data/fire_sequences')
   sequences_dir.mkdir(exist_ok=True)
   
   for _, tile in tiles_df.iterrows():
       for episode_id in tile['fire_episode_ids']:
           process_episode_sequence(tile, episode_id)
   ```

3. **Process Episode Sequence**
   ```python
   def process_episode_sequence(tile, episode_id):
       # Get fire detections in this episode
       ep_fire = df_fire[df_fire['episode_id'] == episode_id].copy()
       
       # Define time window
       t_start = ep_fire['datetime'].min() - pd.Timedelta(hours=TIME_BEFORE_FIRE)
       t_end = ep_fire['datetime'].max() + pd.Timedelta(hours=TIME_AFTER_FIRE)
       
       # Generate timesteps
       timestamps = pd.date_range(t_start, t_end, freq=f'{TIMESTEP_HOURS}H')
       
       # Initialize arrays
       H, W = TILE_SIZE, TILE_SIZE
       T = len(timestamps)
       
       fire_masks = np.zeros((T, H, W), dtype=np.uint8)
       fire_intensity = np.zeros((T, H, W), dtype=np.float32)
       fire_age = np.zeros((T, H, W), dtype=np.float32)
       
       # For each timestep
       for t_idx, timestamp in enumerate(timestamps):
           # Find detections within timestep window
           window_start = timestamp
           window_end = timestamp + pd.Timedelta(hours=TIMESTEP_HOURS)
           
           detections = ep_fire[
               (ep_fire['datetime'] >= window_start) & 
               (ep_fire['datetime'] < window_end)
           ]
           
           if len(detections) == 0:
               continue
           
           # Convert coordinates to tile-local pixel indices
           for _, det in detections.iterrows():
               # Global grid coordinates
               x_global = det['x']
               y_global = det['y']
               
               # Convert to tile-local
               x_local = int((x_global - tile['x_start']))
               y_local = int((y_global - tile['y_start']))
               
               # Bounds check
               if 0 <= x_local < W and 0 <= y_local < H:
                   fire_masks[t_idx, y_local, x_local] = 1
                   fire_intensity[t_idx, y_local, x_local] = det['i']
                   fire_age[t_idx, y_local, x_local] = det['tm']
       
       # Interpolate/spread fire between sparse detections
       fire_masks, fire_intensity, fire_age = interpolate_fire_spread(
           fire_masks, fire_intensity, fire_age, timestamps
       )
       
       # Save sequence
       output_file = sequences_dir / f"tile_{tile['tile_id']}_ep_{episode_id}.npz"
       np.savez_compressed(
           output_file,
           timestamps=timestamps.astype('datetime64[ns]').astype(np.int64),
           fire_masks=fire_masks,
           fire_intensity=fire_intensity,
           fire_age=fire_age,
           tile_bounds=(tile['x_start'], tile['x_end'], 
                        tile['y_start'], tile['y_end']),
           episode_id=episode_id
       )
   ```

4. **Fire Spread Interpolation** (Optional Enhancement)
   ```python
   def interpolate_fire_spread(masks, intensity, age, timestamps):
       """
       Fill gaps between sparse detections using simple spread model
       or just keep sparse (for RL, sparse is okay)
       """
       # Option 1: Keep sparse (simpler)
       return masks, intensity, age
       
       # Option 2: Apply morphological dilation between detections
       # Option 3: Use simple cellular automaton
       # Option 4: Train separate ML model for spread prediction
   ```

#### Output
**Directory**: `../tilled_data/fire_sequences/`

Files: `tile_{tile_id}_ep_{episode_id}.npz`

Contents:
- `timestamps`: (T,) int64 - Unix timestamps in nanoseconds
- `fire_masks`: (T, H, W) uint8 - Binary fire presence
- `fire_intensity`: (T, H, W) float32 - Fire intensity values
- `fire_age`: (T, H, W) float32 - Hours since ignition
- `tile_bounds`: (4,) tuple - Grid coordinates
- `episode_id`: int - Episode identifier

**Summary**: `../tilled_data/sequence_summary.json`
```json
{
  "total_sequences": 567,
  "avg_timesteps": 32.4,
  "total_fire_pixels": 45678,
  "temporal_resolution_hours": 6
}
```

---

### Step 4: Environment Assembly
**Script**: `04_environment_assembly.py`

**Purpose**: Combine all data sources into complete RL environment states

#### Process

1. **Load All Embedded Rasters**
   ```python
   # Static features
   with rasterio.open('../embedded_data/dem_rsp_embedded.tif') as src:
       dem_full = src.read(1)
       rsp_full = src.read(2)
       grid_transform = src.transform
   
   with rasterio.open('../embedded_data/lcm_embedded.tif') as src:
       lcm_full = src.read(1)
   
   with rasterio.open('../embedded_data/fsm_embedded.tif') as src:
       fsm_full = src.read(1)
   
   with rasterio.open('../embedded_data/ndvi_embedded.tif') as src:
       ndvi_full = src.read(1)  # Use most recent band
   
   with rasterio.open('../embedded_data/kma_weather_embedded.tif') as src:
       weather_full = src.read()  # All 5 bands
   ```

2. **For Each Fire Sequence**
   ```python
   envs_dir = Path('../tilled_data/environments')
   envs_dir.mkdir(exist_ok=True)
   
   sequence_files = sorted(Path('../tilled_data/fire_sequences').glob('*.npz'))
   
   for seq_file in tqdm(sequence_files):
       create_environment(seq_file)
   ```

3. **Create Environment**
   ```python
   def create_environment(seq_file):
       # Load sequence
       seq_data = np.load(seq_file)
       fire_masks = seq_data['fire_masks']
       fire_intensity = seq_data['fire_intensity']
       fire_age = seq_data['fire_age']
       timestamps = pd.to_datetime(seq_data['timestamps'])
       tile_bounds = seq_data['tile_bounds']
       episode_id = seq_data['episode_id']
       
       x_start, x_end, y_start, y_end = tile_bounds
       
       # Extract static features for this tile
       static_continuous = np.stack([
           dem_full[y_start:y_end, x_start:x_end],
           rsp_full[y_start:y_end, x_start:x_end],
           ndvi_full[y_start:y_end, x_start:x_end],
           weather_full[0, y_start:y_end, x_start:x_end],  # wind_speed
           weather_full[1, y_start:y_end, x_start:x_end],  # wind_x
           weather_full[2, y_start:y_end, x_start:x_end],  # wind_y
           weather_full[3, y_start:y_end, x_start:x_end],  # humidity
           weather_full[4, y_start:y_end, x_start:x_end],  # precipitation
       ], axis=0).astype(np.float32)  # Shape: (8, H, W)
       
       static_categorical = {
           'lcm': lcm_full[y_start:y_end, x_start:x_end].astype(np.int32),
           'fsm': fsm_full[y_start:y_end, x_start:x_end].astype(np.int32)
       }
       
       # Create environment dict
       env_data = {
           'episode_id': int(episode_id),
           'tile_id': seq_file.stem,
           'timestamps': timestamps,
           'tile_bounds': tile_bounds,
           
           # Static state (never changes during episode)
           'static_continuous': static_continuous,  # (8, H, W)
           'static_categorical': static_categorical,
           
           # Dynamic fire state (changes each timestep)
           'fire_sequence': {
               'masks': fire_masks,         # (T, H, W)
               'intensity': fire_intensity, # (T, H, W)
               'age': fire_age              # (T, H, W)
           },
           
           # Metadata
           'num_timesteps': len(timestamps),
           'spatial_shape': (fire_masks.shape[1], fire_masks.shape[2]),
           'total_fire_detections': int(fire_masks.sum())
       }
       
       # Save as compressed pickle
       output_file = envs_dir / f"{seq_file.stem}.pkl"
       with open(output_file, 'wb') as f:
           pickle.dump(env_data, f, protocol=pickle.HIGHEST_PROTOCOL)
   ```

#### Output
**Directory**: `../tilled_data/environments/`

Files: `tile_{tile_id}_ep_{episode_id}.pkl`

Each contains:
```python
{
    'episode_id': int,
    'tile_id': str,
    'timestamps': pd.DatetimeIndex,
    'tile_bounds': tuple,
    'static_continuous': np.ndarray,  # (8, H, W) float32
    'static_categorical': dict,
    'fire_sequence': dict,
    'num_timesteps': int,
    'spatial_shape': tuple,
    'total_fire_detections': int
}
```

**Manifest**: `../tilled_data/environments/env_manifest.json`
```json
[
  {
    "env_file": "tile_XXX_ep_YYY.pkl",
    "episode_id": 123,
    "num_timesteps": 28,
    "total_fire_pixels": 456,
    "spatial_shape": [256, 256],
    "tile_bounds": [1000, 1256, 2000, 2256],
    "start_time": "2020-03-15T10:00:00",
    "end_time": "2020-03-22T16:00:00"
  },
  ...
]
```

---

### Step 5: Dataset Split
**Script**: `05_dataset_split.py`

**Purpose**: Split environments into train/val/test sets with stratification

#### Configuration
```python
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Stratification criteria
STRATIFY_BY = {
    'fire_size': [0, 100, 500, 2000],  # Fire pixel bins
    'season': ['winter', 'spring', 'summer', 'fall'],
    'duration_hours': [0, 24, 72, 168]
}
```

#### Process

1. **Load Environment Manifest**
   ```python
   with open('../tilled_data/environments/env_manifest.json') as f:
       manifest = json.load(f)
   
   df_envs = pd.DataFrame(manifest)
   ```

2. **Add Stratification Features**
   ```python
   # Season
   df_envs['start_time'] = pd.to_datetime(df_envs['start_time'])
   df_envs['month'] = df_envs['start_time'].dt.month
   df_envs['season'] = df_envs['month'].map({
       12: 'winter', 1: 'winter', 2: 'winter',
       3: 'spring', 4: 'spring', 5: 'spring',
       6: 'summer', 7: 'summer', 8: 'summer',
       9: 'fall', 10: 'fall', 11: 'fall'
   })
   
   # Duration
   df_envs['duration_hours'] = (
       pd.to_datetime(df_envs['end_time']) - df_envs['start_time']
   ).dt.total_seconds() / 3600
   
   # Fire size bin
   df_envs['fire_size_bin'] = pd.cut(
       df_envs['total_fire_pixels'],
       bins=STRATIFY_BY['fire_size'],
       labels=['small', 'medium', 'large']
   )
   
   # Geographic region (based on tile_bounds)
   df_envs['region'] = df_envs['tile_bounds'].apply(infer_region)
   ```

3. **Stratified Split**
   ```python
   from sklearn.model_selection import train_test_split
   
   # First split: train vs. (val + test)
   train_envs, temp_envs = train_test_split(
       df_envs,
       test_size=(VAL_RATIO + TEST_RATIO),
       stratify=df_envs[['season', 'fire_size_bin']],
       random_state=42
   )
   
   # Second split: val vs. test
   val_envs, test_envs = train_test_split(
       temp_envs,
       test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
       stratify=temp_envs[['season', 'fire_size_bin']],
       random_state=42
   )
   ```

4. **Verify Split Quality**
   ```python
   print("Train set:")
   print(train_envs['season'].value_counts())
   print(train_envs['fire_size_bin'].value_counts())
   
   print("\nValidation set:")
   print(val_envs['season'].value_counts())
   
   print("\nTest set:")
   print(test_envs['season'].value_counts())
   ```

5. **Save Splits**
   ```python
   splits_dir = Path('../tilled_data/splits')
   splits_dir.mkdir(exist_ok=True)
   
   train_envs[['env_file', 'episode_id']].to_json(
       splits_dir / 'train_envs.json', orient='records', indent=2
   )
   
   val_envs[['env_file', 'episode_id']].to_json(
       splits_dir / 'val_envs.json', orient='records', indent=2
   )
   
   test_envs[['env_file', 'episode_id']].to_json(
       splits_dir / 'test_envs.json', orient='records', indent=2
   )
   ```

#### Output
**Directory**: `../tilled_data/splits/`

Files:
- `train_envs.json` - List of training environment files
- `val_envs.json` - Validation set
- `test_envs.json` - Test set
- `split_statistics.txt` - Summary statistics

---

### Step 6: Environment Validation
**Script**: `06_environment_validation.py`

**Purpose**: Quality checks and visualization of created environments

#### Process

1. **Load Random Environments**
   ```python
   import pickle
   import random
   
   with open('../tilled_data/splits/train_envs.json') as f:
       train_list = json.load(f)
   
   sample_files = random.sample(train_list, 10)
   ```

2. **Validation Checks**
   ```python
   for env_file in sample_files:
       with open(f'../tilled_data/environments/{env_file["env_file"]}', 'rb') as f:
           env = pickle.load(f)
       
       # Check shapes
       assert env['static_continuous'].shape[0] == 8
       assert env['static_continuous'].shape[1:] == (256, 256)
       
       # Check no NaNs in critical features
       assert not np.isnan(env['static_continuous']).all(axis=(1, 2)).any()
       
       # Check fire sequence consistency
       T = env['num_timesteps']
       assert env['fire_sequence']['masks'].shape[0] == T
       
       # Check categorical classes are valid
       assert env['static_categorical']['lcm'].max() < NUM_LCM_CLASSES
   ```

3. **Visualize Sample Environments**
   ```python
   def visualize_environment(env, output_path):
       fig, axes = plt.subplots(4, 4, figsize=(16, 16))
       
       # Row 1: Static continuous features
       axes[0, 0].imshow(env['static_continuous'][0], cmap='terrain')
       axes[0, 0].set_title('DEM')
       
       axes[0, 1].imshow(env['static_continuous'][1], cmap='RdYlGn')
       axes[0, 1].set_title('RSP')
       
       axes[0, 2].imshow(env['static_continuous'][2], cmap='Greens')
       axes[0, 2].set_title('NDVI')
       
       axes[0, 3].imshow(env['static_continuous'][3], cmap='Blues')
       axes[0, 3].set_title('Wind Speed')
       
       # Row 2: Categorical features
       axes[1, 0].imshow(env['static_categorical']['lcm'], cmap='tab20')
       axes[1, 0].set_title('Land Cover')
       
       axes[1, 1].imshow(env['static_categorical']['fsm'], cmap='tab20')
       axes[1, 1].set_title('Forest Type')
       
       # Row 3-4: Fire progression over time
       T = env['num_timesteps']
       sample_times = np.linspace(0, T-1, 6, dtype=int)
       
       for i, t in enumerate(sample_times):
           row = 2 + i // 3
           col = i % 3
           
           axes[row, col].imshow(
               env['fire_sequence']['masks'][t],
               cmap='Reds',
               alpha=0.7
           )
           axes[row, col].imshow(
               env['static_categorical']['lcm'],
               cmap='Greys',
               alpha=0.3
           )
           axes[row, col].set_title(f"t={t}")
       
       plt.tight_layout()
       plt.savefig(output_path, dpi=150)
       plt.close()
   ```

#### Output
**Directory**: `../tilled_data/validation/`

- Sample visualizations (PNG files)
- `validation_report.txt` - Summary of checks
- `data_statistics.json` - Aggregate statistics

---

## Output Directory Structure

After completing the tilling pipeline:

```
tilled_data/
├── tile_index.parquet           # Spatial tile metadata
├── tile_summary.txt             # Summary statistics
├── episode_flags.csv            # Quality flags per episode
├── cluster_validation_report.txt
├── fire_sequences/              # Temporal fire progressions
│   ├── tile_0_0_ep_123.npz
│   ├── tile_0_128_ep_124.npz
│   └── ...
├── sequence_summary.json
├── environments/                # Complete RL environments
│   ├── tile_0_0_ep_123.pkl
│   ├── tile_0_128_ep_124.pkl
│   ├── ...
│   └── env_manifest.json
├── splits/                      # Train/val/test split
│   ├── train_envs.json
│   ├── val_envs.json
│   ├── test_envs.json
│   └── split_statistics.txt
└── validation/                  # QC visualizations
    ├── sample_env_001.png
    ├── sample_env_002.png
    ├── ...
    ├── validation_report.txt
    └── data_statistics.json
```

---

## Usage Example

### Running the Full Pipeline

```bash
cd tilling_src

# Step 1: Spatial tiling
python 01_spatial_tiling.py
# Expected time: ~5-10 minutes
# Creates: tile_index.parquet

# Step 2: Validate clusters
python 02_fire_cluster_validation.py
# Expected time: ~10-15 minutes
# Updates: tile_index.parquet

# Step 3: Temporal segmentation
python 03_temporal_segmentation.py
# Expected time: ~30-60 minutes
# Creates: fire_sequences/*.npz

# Step 4: Environment assembly
python 04_environment_assembly.py
# Expected time: ~1-2 hours
# Creates: environments/*.pkl

# Step 5: Dataset split
python 05_dataset_split.py
# Expected time: ~1 minute
# Creates: splits/*.json

# Step 6: Validation
python 06_environment_validation.py
# Expected time: ~5-10 minutes
# Creates: validation/*
```

### Or Use the Pipeline Script

```bash
bash run_tilling_pipeline.sh
```

---

## Integration with RL Training

After tilling completes, the RL training code will:

1. Load environment files from `tilled_data/environments/`
2. Use split files from `tilled_data/splits/`
3. Implement `WildfireEnv` gym wrapper that:
   - Loads `.pkl` environment
   - Provides `reset()` and `step()` interface
   - Manages state transitions based on fire sequences
   - Computes rewards based on suppression actions

See `../PLAN.md` Phase 3 for RL training details.

---

## Key Parameters & Tuning

### Tile Size Trade-offs

| Tile Size | Pros | Cons |
|-----------|------|------|
| 128x128 | More samples, faster training | Limited context, small fires only |
| 256x256 | Good balance | Recommended default |
| 512x512 | Full fire context, strategic view | Fewer samples, slower training |

### Temporal Resolution

| Timestep | Pros | Cons |
|----------|------|------|
| 1 hour | Fine-grained dynamics | Many timesteps, sparse detections |
| 6 hours | Balance detail/efficiency | Recommended default |
| 24 hours | Strategic planning | Miss rapid changes |

### Overlap (Stride)

| Stride | Overlap | Effect |
|--------|---------|--------|
| = Tile Size | 0% | No overlap, fewer samples |
| = Tile Size / 2 | 50% | 4x more samples |
| = Tile Size / 4 | 75% | 16x more samples |

---

## Troubleshooting

### Issue: Too Few Tiles Generated
- **Cause**: MIN_FIRE_EPISODES too high
- **Fix**: Lower to 1, or adjust TILE_SIZE

### Issue: Environments Too Large
- **Cause**: TILE_SIZE too big
- **Fix**: Reduce to 128 or 192

### Issue: Fire Sequences Sparse
- **Cause**: TIMESTEP_HOURS too small
- **Fix**: Increase to 12 or 24 hours

### Issue: Memory Error During Assembly
- **Cause**: Loading all rasters at once
- **Fix**: Process in batches, use memory-mapped arrays

---

## Next Steps

Once tilling is complete:

1. **Verify output**: Check `validation/` directory
2. **Inspect samples**: Load a few `.pkl` files in Jupyter
3. **Proceed to RL training**: See `../PLAN.md` Phase 3
4. **Implement gym environment**: Create `WildfireEnv` wrapper

---

**Last Updated**: November 4, 2025  
**Status**: Ready for implementation

