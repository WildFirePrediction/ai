"""
Fire Episode Embedding Pipeline
Creates 30x30 grid embeddings for each fire episode with:
- DEM data (2 channels: elevation, slope)
- KMA weather data (9 channels)
Total: 11 channels

Grid is centered on fire centroid, 400m resolution (12km x 12km coverage)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from pyproj import Transformer
import rasterio
from rasterio.transform import from_bounds
from scipy.interpolate import griddata
import os

# Coordinate transformer: WGS84 -> EPSG:5179 (Korean projection)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)

GRID_SIZE = 30
CELL_SIZE = 400  # meters
HALF_EXTENT = (GRID_SIZE * CELL_SIZE) / 2  # 6000m = 6km radius

def load_dem_data():
    """Load DEM raster"""
    dem_path = Path("data/DigitalElevationModel/90m_GRS80.tif")
    return rasterio.open(dem_path)

def load_station_coords():
    """Load KMA weather station coordinates"""
    stations = {}
    station_file = Path("data/KMA/kma_weather_station_5179.csv")
    df = pd.read_csv(station_file)
    for _, row in df.iterrows():
        # Already in EPSG:5179
        stations[row['STN']] = (row['X'], row['Y'])
    return stations

def calculate_slope(elevation_grid):
    """Calculate slope from elevation grid"""
    dy, dx = np.gradient(elevation_grid, CELL_SIZE)
    slope = np.sqrt(dx**2 + dy**2)
    return slope

def embed_dem(centroid_x, centroid_y, dem_raster):
    """Embed DEM data for 30x30 grid centered on fire"""
    from rasterio.windows import from_bounds as window_from_bounds
    from rasterio.enums import Resampling
    
    # Define grid bounds
    x_min = centroid_x - HALF_EXTENT
    x_max = centroid_x + HALF_EXTENT
    y_min = centroid_y - HALF_EXTENT
    y_max = centroid_y + HALF_EXTENT
    
    # Read DEM window and resample to 30x30
    try:
        window = window_from_bounds(x_min, y_min, x_max, y_max, dem_raster.transform)
        elevation = dem_raster.read(1, window=window, 
                                   out_shape=(GRID_SIZE, GRID_SIZE),
                                   resampling=Resampling.bilinear)
    except:
        elevation = np.zeros((GRID_SIZE, GRID_SIZE))
    
    # Calculate slope
    slope = calculate_slope(elevation)
    
    # Normalize
    elevation_norm = (elevation - elevation.mean()) / (elevation.std() + 1e-8)
    slope_norm = (slope - slope.mean()) / (slope.std() + 1e-8)
    
    return np.stack([elevation_norm, slope_norm], axis=0)  # (2, 30, 30)

def embed_weather(centroid_x, centroid_y, timestamp_dir, station_coords):
    """Embed KMA weather data for 30x30 grid"""
    # Define grid bounds
    x_min = centroid_x - HALF_EXTENT
    x_max = centroid_x + HALF_EXTENT
    y_min = centroid_y - HALF_EXTENT
    y_max = centroid_y + HALF_EXTENT
    
    # Create target grid
    x_coords = np.linspace(x_min, x_max, GRID_SIZE)
    y_coords = np.linspace(y_max, y_min, GRID_SIZE)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    target_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    # Load weather data
    csv_file = list(timestamp_dir.glob("*.csv"))[0]
    df = pd.read_csv(csv_file)
    
    # KMA CSV columns: TA (temp), HM (humidity), WS1 (wind speed), WD1 (wind dir),
    # RN-15m (precip), PA (pressure), TD (dew point)
    # Missing: cloud_cover, visibility - use zeros
    feature_mapping = {
        'TA': 'temperature',
        'HM': 'humidity', 
        'WS1': 'wind_speed',
        'WD1': 'wind_direction',
        'RN-15m': 'precipitation',
        'PA': 'atmospheric_pressure',
        'TD': 'dew_point'
    }
    
    weather_grids = []
    for kma_col, feature_name in feature_mapping.items():
        # Get station values
        station_values = []
        station_points = []
        
        for _, row in df.iterrows():
            if row['STN'] in station_coords and row[kma_col] > -99:  # Filter invalid data
                x, y = station_coords[row['STN']]
                station_points.append([x, y])
                station_values.append(row[kma_col])
        
        if len(station_values) < 4:
            # Not enough points for triangulation - use mean or zeros
            if len(station_values) > 0:
                mean_val = np.mean(station_values)
                weather_grids.append(np.full((GRID_SIZE, GRID_SIZE), mean_val))
            else:
                weather_grids.append(np.zeros((GRID_SIZE, GRID_SIZE)))
            continue
        
        station_points = np.array(station_points)
        station_values = np.array(station_values)
        
        # Interpolate to grid
        grid_values = griddata(station_points, station_values, target_points, 
                              method='linear', fill_value=station_values.mean())
        grid = grid_values.reshape(GRID_SIZE, GRID_SIZE)
        
        # Normalize
        grid_norm = (grid - grid.mean()) / (grid.std() + 1e-8)
        weather_grids.append(grid_norm)
    
    # Add cloud_cover and visibility as zeros (not available)
    weather_grids.append(np.zeros((GRID_SIZE, GRID_SIZE)))
    weather_grids.append(np.zeros((GRID_SIZE, GRID_SIZE)))
    
    return np.stack(weather_grids, axis=0)  # (9, 30, 30)

def process_fire_episode(episode_id, episode_df, dem_raster, station_coords, output_dir):
    """Process single fire episode - create embeddings for sampled timestamps (10 per episode)"""
    # Calculate centroid in EPSG:5179
    centroid_x = episode_df['x'].mean()
    centroid_y = episode_df['y'].mean()
    
    # Get unique timestamps and sample 10 evenly spaced
    all_timestamps = sorted(episode_df['datetime'].unique())
    if len(all_timestamps) <= 10:
        timestamps = all_timestamps
    else:
        # Sample 10 evenly spaced timestamps
        indices = np.linspace(0, len(all_timestamps)-1, 10, dtype=int)
        timestamps = [all_timestamps[i] for i in indices]
    
    print(f"{len(timestamps)}/{len(all_timestamps)} timestamps...", flush=True, end=' ')
    
    episode_data = []
    for ts_idx, timestamp in enumerate(timestamps):
        ts_str = pd.to_datetime(timestamp).strftime('%Y%m%d%H%M')
        kma_dir = Path(f"data/KMA/{ts_str}")
        
        if not kma_dir.exists():
            continue
        
        # Embed DEM (static, same for all timestamps)
        dem_data = embed_dem(centroid_x, centroid_y, dem_raster)
        
        # Embed weather (dynamic per timestamp)
        weather_data = embed_weather(centroid_x, centroid_y, kma_dir, station_coords)
        
        # Get fire mask for this timestamp
        ts_fires = episode_df[episode_df['datetime'] == timestamp]
        fire_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        
        x_min = centroid_x - HALF_EXTENT
        y_max = centroid_y + HALF_EXTENT
        
        for _, fire in ts_fires.iterrows():
            # Convert fire position to grid indices
            i = int((y_max - fire['y']) / CELL_SIZE)
            j = int((fire['x'] - x_min) / CELL_SIZE)
            if 0 <= i < GRID_SIZE and 0 <= j < GRID_SIZE:
                fire_mask[i, j] = 1.0
        
        # State: (11, 30, 30) - DEM(2) + Weather(9)
        # Fire mask separate: (30, 30)
        state = np.concatenate([dem_data, weather_data], axis=0)
        
        episode_data.append({
            'timestamp': timestamp,
            'state': state,  # (11, 30, 30)
            'fire_mask': fire_mask,  # (30, 30)
            'centroid': (centroid_x, centroid_y)
        })
    
    # Save episode data
    if len(episode_data) > 0:
        output_file = output_dir / f"episode_{episode_id:03d}.npz"
        states_array = np.array([d['state'] for d in episode_data])  # (T, 11, 30, 30)
        fire_masks = np.array([d['fire_mask'] for d in episode_data])  # (T, 30, 30)
        np.savez_compressed(
            output_file,
            states=states_array,
            fire_masks=fire_masks,
            timestamps=np.array([d['timestamp'] for d in episode_data]),
            centroid=np.array(episode_data[0]['centroid'])
        )
        return True
    return False

def main():
    print("=" * 80)
    print("Fire Episode Embedding Pipeline - 11 Channels (DEM + Weather)")
    print("=" * 80)
    
    # Load resources
    print("\nLoading DEM data...")
    dem_raster = load_dem_data()
    
    print("Loading weather stations...")
    station_coords = load_station_coords()
    print(f"Loaded {len(station_coords)} stations")
    
    # Load fire data
    print("\nLoading fire episodes...")
    fire_df = pd.read_parquet("data/filtered_fires/filtered_viirs.parquet")
    n_episodes = fire_df['episode_id'].max() + 1
    print(f"Found {n_episodes} episodes")
    
    # Create output directory
    output_dir = Path("embedded_data/fire_episodes_11ch")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each episode
    print("\nProcessing episodes...")
    print("=" * 80)
    
    success_count = 0
    for episode_id in range(n_episodes):
        episode_df = fire_df[fire_df['episode_id'] == episode_id]
        if len(episode_df) == 0:
            continue
        
        print(f"Processing episode {episode_id}... ", flush=True, end='')
        success = process_fire_episode(episode_id, episode_df, dem_raster, 
                                      station_coords, output_dir)
        if success:
            success_count += 1
            print(f"SUCCESS ({success_count}/{n_episodes})", flush=True)
        else:
            print("SKIP", flush=True)
    
    print("=" * 80)
    print(f"\nCompleted: {success_count} episodes embedded successfully")
    print(f"Output directory: {output_dir}")
    
    dem_raster.close()

if __name__ == "__main__":
    main()
