# Data Embedding Pipeline - Technical Documentation

## Overview

This pipeline processes multi-source geospatial data for wildfire prediction RL training. It transforms heterogeneous data into a unified 400m grid (EPSG:5179) with normalized features ready for A3C model input.

---

## Architecture

### Pipeline Flow

```
Raw Data (data/) 
    ↓
[01] NASA VIIRS → Fire detections with pre-clustered episodes
    ↓
[02] DEM/RSP → Establishes 400m grid baseline
    ↓
[03] LCM → Land cover rasterization
    ↓
[04] FSM → Forest stand rasterization
    ↓
[05] NDVI → Vegetation index processing
    ↓
[06] KMA → Weather data interpolation
    ↓
[07] Final → State vector composition
    ↓
Embedded Data (embedded_data/)
    ↓
state_vectors.npz (Ready for RL)
```

### Data Sources

| Source | Type | Resolution | Purpose |
|--------|------|------------|---------|
| NASA VIIRS | Point | Variable | Fire hotspots with episodes |
| DEM | Raster | 90m | Elevation |
| RSP | Raster | Variable | Relative slope position |
| LCM | Vector | N/A | Land cover classification |
| FSM | Vector | N/A | Forest stand types (임상도) |
| NDVI | Raster | Variable | Vegetation index |
| KMA | Point | Station | Weather observations |

---

## File Structure

### Input Data Structure

```
data/
├── NASA/VIIRS/
│   └── DL_FIRE_YYYYMMDD-YYYYMMDD/
│       └── clustered_fire_archive_*.csv
├── DigitalElevationModel/
│   └── 90m_GRS80.tif
├── RelativeSlopePosition/
│   └── *.tif
├── LandCoverMap/
│   └── SG05_*/
│       └── *.shp
├── ForestStandMap/
│   └── [province]/
│       └── 51_*.shp
├── NDVI/
│   └── *.tif
└── KMA/
    └── YYYYMMDDHHMM/
        └── AWS_*.csv
```

### Output Structure

```
embedded_data/
├── nasa_viirs_embedded.parquet          # Fire point data
├── nasa_viirs_episode_index.parquet     # Episode metadata
├── nasa_viirs_norm_stats.json           # Normalization stats
├── dem_rsp_embedded.tif                 # 2 bands: DEM, RSP
├── dem_rsp_norm_stats.json              # Grid configuration
├── lcm_embedded.tif                     # Land cover raster
├── lcm_class_mapping.json               # Class ID mappings
├── fsm_embedded.tif                     # Forest stand raster
├── fsm_class_mapping.json               # Type ID mappings
├── ndvi_embedded.tif                    # NDVI (1+ bands)
├── ndvi_norm_stats.json                 # NDVI statistics
├── kma_weather_embedded.tif             # 5 bands: w, d_x, d_y, rh, r
├── kma_weather_norm_stats.json          # Weather stats
├── state_vectors.npz                    # Final state tensor
└── grid_metadata.json                   # Complete metadata
```

---

## Script Details

### 01_nasa_viirs_embedding.py

**Purpose:** Process NASA VIIRS fire hotspot data with pre-clustered episodes

**Input:** `data/NASA/VIIRS/*/clustered_fire_archive_*.csv`

**Processing:**
1. Load pre-clustered fire archives (no additional clustering needed)
2. Filter low confidence detections (confidence != 'l')
3. Transform coordinates from WGS84 (EPSG:4326) to Korean TM (EPSG:5179)
4. Extract temperature from BRIGHT_T31: `te = BRIGHT_T31 - 273.15`
5. Compute fire intensity: `i = α * FRP + β * (BRIGHTNESS - 273.15)`, then `i' = log(1 + i)`
6. Calculate time elapsed within each episode: `tm = (datetime - episode_start) / 3600`
7. Normalize all features using z-score: `(x - μ) / σ`

**Output:**
- `nasa_viirs_embedded.parquet` - All fire detections with normalized features
- `nasa_viirs_episode_index.parquet` - Episode-level statistics
- `nasa_viirs_norm_stats.json` - Normalization parameters

**Key Features:**
- Uses NASA's pre-clustered episodes (CLUSTER_ID column)
- No DBSCAN needed - faster and validated
- Episode statistics: duration, detection count, intensity

---

### 02_dem_rsp_embedding.py

**Purpose:** Process elevation and slope data, establish grid baseline

**Input:** 
- `data/DigitalElevationModel/90m_GRS80.tif`
- `data/RelativeSlopePosition/*.tif`

**Processing:**
1. Load DEM raster
2. Normalize DEM: `dem_norm = (dem - dem_min) / (dem_max - dem_min)`
3. Load RSP raster (or create dummy if missing)
4. Define target grid:
   - Resolution: 400m × 400m
   - CRS: EPSG:5179
   - Snap origin to 400m grid
5. Reproject both DEM and RSP to target grid using bilinear interpolation

**Output:**
- `dem_rsp_embedded.tif` - 2 bands: dem_norm, rsp_norm
- `dem_rsp_norm_stats.json` - Grid configuration for subsequent scripts

**Grid Parameters:**
```python
x0 = floor(xmin / 400) * 400
y0 = floor(ymin / 400) * 400
width = ceil((xmax - x0) / 400)
height = ceil((ymax - y0) / 400)
```

---

### 03_lcm_embedding.py

**Purpose:** Rasterize Land Cover Map shapefiles to grid

**Input:** `data/LandCoverMap/*/*.shp`

**Processing:**
1. Load grid configuration from `dem_rsp_norm_stats.json`
2. Load and combine all LCM shapefiles
3. Reproject to EPSG:5179 if needed
4. Identify land cover class column
5. Create integer encoding: `{class_name: class_id}`
6. Rasterize to 400m grid using `rasterio.features.rasterize`
7. Save class mapping for embedding layer

**Output:**
- `lcm_embedded.tif` - Rasterized land cover class IDs
- `lcm_class_mapping.json` - Class name ↔ ID mappings

**PyTorch Usage:**
```python
lcm_embed = nn.Embedding(num_classes, embedding_dim=16)
lcm_features = lcm_embed(lcm_ids)  # (H, W, 16)
```

---

### 04_fsm_embedding.py

**Purpose:** Rasterize Forest Stand Map (임상도) from all provinces

**Input:** `data/ForestStandMap/*/51_*.shp`

**Processing:**
1. Load grid configuration
2. Load shapefiles from all province directories
3. Combine into single GeoDataFrame
4. Identify forest type column (임상, IMSNAG, etc.)
5. Create integer encoding for forest types
6. Rasterize to 400m grid
7. Compute coverage statistics

**Output:**
- `fsm_embedded.tif` - Rasterized forest type IDs
- `fsm_class_mapping.json` - Type name ↔ ID mappings

**Forest Type Categories:**
- Coniferous forests (침엽수림)
- Deciduous forests (활엽수림)
- Mixed forests (혼효림)
- Other vegetation types

---

### 05_ndvi_embedding.py

**Purpose:** Process NDVI vegetation index data

**Input:** `data/NDVI/*.tif`

**Processing:**
1. Load grid configuration
2. Load NDVI raster files (multiple time periods if available)
3. Normalize NDVI from [-1, 1] to [0, 1]: `ndvi_norm = (ndvi + 1) / 2`
4. Reproject to 400m grid
5. If multiple time periods: compute temporal mean and std
6. Save as multi-band raster (each band = time period)

**Output:**
- `ndvi_embedded.tif` - 1+ bands of normalized NDVI
- `ndvi_norm_stats.json` - NDVI statistics

**Temporal Handling:**
- Single file: Use as-is
- Multiple files: Save as bands, compute temporal mean
- Bi-weekly updates: Up to 26 bands per year

---

### 06_kma_weather_embedding.py

**Purpose:** Interpolate weather station data to grid

**Input:** `data/KMA/*/AWS_*.csv`

**Processing:**
1. Load grid configuration
2. Load KMA AWS weather data
3. Transform station coordinates to EPSG:5179
4. Decompose wind direction: `d_x = sin(direction)`, `d_y = cos(direction)`
5. Interpolate variables to grid using `scipy.interpolate.griddata`:
   - Wind speed (w)
   - Wind direction components (d_x, d_y)
   - Relative humidity (rh)
   - Precipitation (r)
6. Normalize all variables using z-score

**Output:**
- `kma_weather_embedded.tif` - 5 bands: w, d_x, d_y, rh, r
- `kma_weather_norm_stats.json` - Weather normalization parameters

**Note:** Currently uses synthetic station locations. Replace with actual coordinates:
```python
stations = pd.read_csv('../data/KMA/station_info.csv')
df_kma = df_kma.merge(stations, on='station_id')
```

---

### 07_final_state_composition.py

**Purpose:** Combine all embedded data into final state vectors

**Input:** All outputs from scripts 01-06

**Processing:**
1. Load all embedded data sources
2. Verify spatial alignment (all grids same shape)
3. Stack continuous features into (C, H, W) tensor
4. Store categorical features separately
5. Create fire episode index from NASA data
6. Save as compressed NumPy archive

**Output:**
- `state_vectors.npz` - Main output with all features
- `episode_index.parquet` - Fire episode metadata
- `grid_metadata.json` - Complete grid and feature information

**State Vector Structure:**
```python
{
    'continuous_features': (13, H, W) float32,
    'feature_names': array(['dem_norm', 'rsp_norm', ...]),
    'lcm_classes': (H, W) uint16,
    'fsm_classes': (H, W) uint16
}
```

---

## Feature Specifications

### Continuous Features (13 total)

| Feature | Source | Range | Normalization | Description |
|---------|--------|-------|---------------|-------------|
| x_norm | NASA | ℝ | z-score | Easting coordinate |
| y_norm | NASA | ℝ | z-score | Northing coordinate |
| te_norm | NASA | ℝ | z-score | Fire temperature (°C) |
| i_norm | NASA | ℝ | z-score | Fire intensity (log-scaled) |
| tm_norm | NASA | ℝ | z-score | Time elapsed (hours, log) |
| dem_norm | DEM | [0,1] | min-max | Elevation |
| rsp_norm | RSP | [0,1] | raw | Relative slope position |
| ndvi_norm | NDVI | [0,1] | [-1,1]→[0,1] | Vegetation index |
| w_norm | KMA | ℝ | z-score | Wind speed |
| d_x_norm | KMA | [-1,1] | raw | Wind direction X |
| d_y_norm | KMA | [-1,1] | raw | Wind direction Y |
| rh_norm | KMA | ℝ | z-score | Relative humidity |
| r_norm | KMA | ℝ | z-score (log) | Precipitation |

### Categorical Features (2 total)

| Feature | Source | Type | Classes | Embedding |
|---------|--------|------|---------|-----------|
| lcm | LCM | uint16 | ~10-50 | 16-dim |
| fsm | FSM | uint16 | ~10-30 | 16-dim |

---

## RL State Definition

### MDP State Components

Based on DATACONFIG.md specification:

```
s = (x, y, te, l, w, d, rh, i, tm, r, dem, rsp, fsm, ndvi)
```

**Mapped to outputs:**
- x, y → x_norm, y_norm (spatial)
- te → te_norm (fire temperature)
- l → lcm (land cover, categorical)
- w, d → w_norm, d_x_norm, d_y_norm (wind)
- rh → rh_norm (humidity)
- i → i_norm (fire intensity)
- tm → tm_norm (time elapsed)
- r → r_norm (precipitation)
- dem → dem_norm (elevation)
- rsp → rsp_norm (slope position)
- fsm → fsm (forest stand, categorical)
- ndvi → ndvi_norm (vegetation)

### Action Space

```
A = {N, NE, NW, S, SE, SW, E, W, NONE}
```

9 discrete actions for fire spread direction.

---

## 🔬 Technical Details

### Coordinate Systems

**Input CRS:**
- NASA VIIRS: EPSG:4326 (WGS84)
- DEM/RSP: Various (auto-detected)
- LCM/FSM: Various (auto-detected)
- KMA: EPSG:4326 (station coordinates)

**Output CRS:**
- All data: EPSG:5179 (Korean 2000 / Unified CS)

**Transformation:**
```python
from pyproj import transform, Proj
proj_src = Proj("EPSG:4326")
proj_dst = Proj("EPSG:5179")
x, y = transform(proj_src, proj_dst, lon, lat)
```

### Grid Specification

**Resolution:** 400m × 400m tiles

**Origin:** Snapped to 400m grid
```python
x0 = floor(xmin / 400) * 400
y0 = floor(ymin / 400) * 400
```

**Extent:** Covers South Korea
- Typical dimensions: 2000-4000 × 2000-4000 pixels
- Total area: ~800-1600 × 800-1600 km

**Transform:**
```
[400,   0, x0,
   0, -400, y1,
   0,   0,  1]
```

### Normalization Methods

**Z-score (continuous):**
```python
x_norm = (x - μ) / σ
```
Used for: coordinates, temperature, intensity, time, wind, humidity, precipitation

**Min-max (bounded):**
```python
x_norm = (x - x_min) / (x_max - x_min)
```
Used for: elevation (DEM)

**Range transformation:**
```python
ndvi_norm = (ndvi + 1) / 2  # [-1,1] → [0,1]
```
Used for: NDVI

**Log scaling:**
```python
i' = log(1 + i)
```
Used for: fire intensity, precipitation (before z-score)

---

## Data Flow Details

### Fire Episode Clustering

**Pre-clustered by NASA:**
- Spatiotemporal DBSCAN already applied
- Parameters: ΔS ≤ 2km, Δt ≤ 168h (7 days)
- CLUSTER_ID column in `clustered_fire_archive_*.csv`

**Episode Characteristics:**
- Unique episode_id for each fire event
- Multiple detections per episode
- Temporal progression tracked
- Spatial extent computed

**Time Elapsed Calculation:**
```python
def calculate_elapsed_time(group):
    t0 = group['datetime'].min()
    group['tm'] = (group['datetime'] - t0).dt.total_seconds() / 3600.0
    return group
```

### Spatial Interpolation

**Method:** Linear interpolation with scipy.griddata

**Applied to:** KMA weather station data

**Process:**
1. Station coordinates → grid coordinates
2. Create meshgrid of target locations
3. Interpolate each variable independently
4. Fill outside convex hull with mean value

**Code:**
```python
grid_values = griddata(
    station_coords,
    values,
    (X_grid, Y_grid),
    method='linear',
    fill_value=np.nanmean(values)
)
```

### Rasterization

**Method:** rasterio.features.rasterize

**Applied to:** LCM and FSM shapefiles

**Parameters:**
- all_touched=True (include boundary pixels)
- dtype='uint16' (support 65k classes)
- fill=0 (no-data value)

**Process:**
```python
shapes = [(geom, class_id) for geom, class_id in zip(geometries, ids)]
raster = rasterize(
    shapes,
    out_shape=(height, width),
    transform=target_transform,
    fill=0,
    dtype='uint16',
    all_touched=True
)
```

---

## Performance & Optimization

### Memory Management

**Large Datasets:**
- Process shapefiles in chunks if memory limited
- Use compressed file formats (Parquet, LZW GeoTIFF)
- Stream data when possible

**Tips:**
```python
# Limit files processed
shp_files = list(dir.rglob('*.shp'))[:N]

# Increase grid resolution (fewer pixels)
target_resolution = 800  # instead of 400

# Use data types efficiently
dem_norm.astype('float32')  # not float64
```

### Processing Time

**Factors:**
- Number of fire detections
- Number of shapefiles
- Grid dimensions
- CPU/disk speed

**Typical Times (Ryzen 9 9950X):**
- NASA: 10-20 min
- DEM/RSP: 5-10 min
- LCM: 20-40 min
- FSM: 30-60 min
- NDVI: 5-10 min
- KMA: 10-15 min
- Final: 5-10 min

**Total:** 1.5-3 hours

### Parallelization

**CPU:**
- DBSCAN: n_jobs=-1 (all cores)
- File loading: Process multiple files in parallel
- Rasterization: Single-threaded (GIL-bound)

**GPU:**
- Not used in embedding pipeline
- Reserved for RL model training

---

## Validation & Quality Control

### Automated Checks

**In each script:**
- File existence verification
- Data type validation
- Range checking
- Shape alignment
- CRS consistency

**Example:**
```python
assert dem_norm.min() >= 0 and dem_norm.max() <= 1, "DEM normalization failed"
assert lcm_raster.shape == (height, width), "Shape mismatch"
```

### Manual Verification

**Fire Episodes:**
```python
import pandas as pd
episodes = pd.read_parquet('embedded_data/episode_index.parquet')
print(f"Episodes: {len(episodes)}")
print(f"Avg duration: {episodes['duration_hours'].mean():.1f}h")
print(f"Episodes >10 detections: {(episodes['num_detections']>10).sum()}")
```

**Grid Alignment:**
```python
import rasterio
files = ['dem_rsp', 'lcm', 'fsm', 'ndvi', 'kma_weather']
for f in files:
    with rasterio.open(f'embedded_data/{f}_embedded.tif') as src:
        print(f"{f}: {src.shape} - {src.crs}")
```

**Feature Statistics:**
```python
import numpy as np
data = np.load('embedded_data/state_vectors.npz')
for i, name in enumerate(data['feature_names']):
    f = data['continuous_features'][i]
    print(f"{name}: μ={f.mean():.3f}, σ={f.std():.3f}")
```

---

## Customization & Extension

### Modify Grid Resolution

**In 02_dem_rsp_embedding.py:**
```python
target_resolution = 800  # Change from 400 to 800m
```

**Impact:**
- Smaller output files
- Faster processing
- Less spatial detail
- Must rerun all subsequent scripts

### Add New Data Source

**Create new script `0X_newsource_embedding.py`:**
1. Load grid configuration from `dem_rsp_norm_stats.json`
2. Process your data
3. Align to target grid (reproject/interpolate)
4. Normalize features
5. Save as GeoTIFF or other format
6. Update `07_final_state_composition.py` to include new source

### Adjust Normalization

**Per-feature basis:**
```python
# Change from z-score to min-max
x_norm = (x - x.min()) / (x.max() - x.min())

# Apply different transformations
x_log = np.log1p(x)
x_sqrt = np.sqrt(x)
```

**Remember to:**
- Update normalization stats JSON
- Document changes
- Update DATACONFIG.md if needed

### Filter Fire Episodes

**In 01_nasa_viirs_embedding.py:**
```python
# After clustering, filter episodes
episodes = episodes[episodes['duration_hours'] >= min_hours]
episodes = episodes[episodes['num_detections'] >= min_count]

# Re-filter fire detections
df_filtered = df_filtered[df_filtered['episode_id'].isin(episodes['episode_id'])]
```

---

## Dependencies

### Core Libraries

```
numpy>=1.24.0          # Array operations
pandas>=2.0.0          # Data manipulation
scipy>=1.10.0          # Interpolation
rasterio>=1.3.0        # Raster I/O
geopandas>=0.13.0      # Vector I/O
pyproj>=3.5.0          # Coordinate transformation
```

### Visualization

```
matplotlib>=3.7.0      # Plotting
seaborn>=0.12.0        # Statistical plots
```

### Utilities

```
tqdm>=4.65.0          # Progress bars
pyarrow>=12.0.0       # Parquet support
```

### System

```
GDAL>=3.6.0           # Geospatial library
```

**Installation:**
```bash
pip install -r requirements_embedding.txt
```

---

## Data Quality & Assumptions

### Data Quality Standards

**NASA VIIRS:**
- Confidence ≥ nominal (filtered: 'l' = low)
- Pre-clustered by NASA experts
- Global coverage, 375m native resolution

**DEM:**
- 90m resolution, adequate for 400m grid
- Consistent across South Korea
- Elevation range: 0-2000m typical

**LCM/FSM:**
- Government-maintained datasets
- Regular updates
- High spatial accuracy

**KMA:**
- Hourly observations
- Quality-controlled
- Sparse spatial coverage (interpolation needed)

**NDVI:**
- Bi-weekly updates (optimal)
- Cloud-free composites
- Range: [-1, 1] typical

### Assumptions

**Spatial:**
- Homogeneity within 400m pixels
- Linear interpolation adequate for weather
- Shapefile boundaries accurate

**Temporal:**
- Fire episode clustering captures all events
- Weather data representative of 400m area
- NDVI temporal mean represents season

**Normalization:**
- Z-score assumes normal distribution (approximate)
- Min-max appropriate for bounded variables
- Log-scaling stabilizes skewed distributions

---

## Production Deployment

### System Requirements

**Minimum:**
- CPU: 4 cores, 2.5+ GHz
- RAM: 16 GB
- Disk: 20 GB free
- OS: Ubuntu 20.04+, Python 3.10+

**Recommended:**
- CPU: 8+ cores, 3.5+ GHz (Ryzen 9 9950X)
- RAM: 32-64 GB
- Disk: SSD, 50+ GB free
- GPU: Optional (for RL training, not embedding)

### Monitoring

**Resource Usage:**
```bash
# Watch resources during processing
watch -n 5 'free -h; echo; df -h .'
```

**Log Files:**
```bash
# Tail logs in real-time
tail -f logs/*.log
```

**Progress:**
- Scripts print progress bars (tqdm)
- Stage completion indicated
- File sizes reported

### Error Handling

**Each script:**
- Validates inputs before processing
- Handles missing data gracefully
- Logs errors with context
- Exits cleanly on failure

**Recovery:**
- Re-run failed script after fixing issue
- Outputs are idempotent (safe to re-run)
- Incremental processing where possible

---

## Output Specifications

### File Formats

**GeoTIFF (.tif):**
- Multi-band rasters
- LZW compression
- Tiled (256×256)
- Float32 or UInt16

**Parquet (.parquet):**
- Columnar format
- Compressed
- Fast I/O
- Pandas-compatible

**NPZ (.npz):**
- Compressed NumPy archive
- Multiple arrays
- Fast loading
- NumPy/PyTorch-compatible

**JSON (.json):**
- Metadata and mappings
- Human-readable
- Easy to parse

### Metadata Standards

**All outputs include:**
- Creation timestamp
- Source data references
- Processing parameters
- Coordinate system
- Data ranges
- Normalization parameters

**Example (grid_metadata.json):**
```json
{
  "width": 3000,
  "height": 3500,
  "tile_size": 400,
  "crs": "EPSG:5179",
  "continuous_feature_names": [...],
  "categorical_features": {...}
}
```

---

## References

### Data Sources

- **NASA FIRMS:** https://firms.modaps.eosdis.nasa.gov/
- **KMA Data Portal:** https://data.kma.go.kr/
- **National Geographic Information:** http://www.ngii.go.kr/

### Standards

- **EPSG:5179:** Korean 2000 / Unified CS
- **GeoTIFF:** https://www.ogc.org/standards/geotiff
- **Parquet:** https://parquet.apache.org/

### Libraries

- **Rasterio:** https://rasterio.readthedocs.io/
- **GeoPandas:** https://geopandas.org/
- **PyProj:** https://pyproj4.github.io/pyproj/

---

**Version:** 1.0  
**Last Updated:** 2025-11-04  
**Status:** Production Ready  
**Environment:** Ubuntu 24.04, Python 3.10+, CUDA 12.x

