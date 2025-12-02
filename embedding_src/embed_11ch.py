"""
Embed fire episodes with 11 channels: 2 DEM + 9 KMA Weather
Creates fire_episodes_11ch directory with episode NPZ files
"""
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from pyproj import Transformer
import os

# Coordinate transformer: WGS84 -> EPSG:5179
transformer = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)

def load_dem_data():
    """Load DEM elevation raster"""
    dem_elev = rasterio.open('data/DigitalElevationModel/90m_GRS80.tif')
    return dem_elev

def calculate_slope(elev):
    """Calculate slope from elevation values (simple gradient)"""
    from scipy.ndimage import sobel
    dx = sobel(elev, axis=1, mode='constant')
    dy = sobel(elev, axis=0, mode='constant')
    slope = np.sqrt(dx**2 + dy**2)
    return slope

def load_kma_stations():
    """Load KMA station coordinates"""
    stations = pd.read_csv('data/KMA/kma_weather_station_5179.csv')
    return stations

def get_dem_value(raster, x, y):
    """Extract DEM value at EPSG:5179 coordinate"""
    try:
        row, col = raster.index(x, y)
        if 0 <= row < raster.height and 0 <= col < raster.width:
            val = raster.read(1, window=((row, row+1), (col, col+1)))[0, 0]
            if np.isfinite(val):
                return val
    except:
        pass
    return np.nan

def load_kma_weather(timestamp_dir, stations):
    """Load KMA weather data for a timestamp"""
    csv_file = timestamp_dir / 'weather.csv'
    if not csv_file.exists():
        return None
    
    df = pd.read_csv(csv_file)
    
    # Merge with station coordinates (match column names)
    weather = pd.merge(df, stations[['STN', 'X', 'Y']], left_on='stn_id', right_on='STN', how='inner')
    
    weather['x'] = weather['X']
    weather['y'] = weather['Y']
    
    return weather

def interpolate_weather(weather_df, grid_x, grid_y):
    """Inverse distance weighting interpolation for 9 weather channels"""
    # Weather channels: temp, humidity, wind_speed, wind_dir, precip, pressure, cloud, visibility, dew_point
    channels = ['temperature', 'humidity', 'wind_speed', 'wind_direction', 
                'precipitation', 'pressure', 'cloud_cover', 'visibility', 'dew_point']
    
    H, W = grid_x.shape
    weather_data = np.zeros((9, H, W), dtype=np.float32)
    
    if weather_df is None or len(weather_df) == 0:
        return weather_data
    
    # IDW interpolation for each cell
    for i in range(H):
        for j in range(W):
            x, y = grid_x[i, j], grid_y[i, j]
            
            # Calculate distances to all stations
            dx = weather_df['x'].values - x
            dy = weather_df['y'].values - y
            distances = np.sqrt(dx**2 + dy**2)
            
            # Avoid division by zero
            distances = np.maximum(distances, 1.0)
            
            # IDW weights (power=2)
            weights = 1.0 / (distances ** 2)
            weights /= weights.sum()
            
            # Weighted average for each channel
            for ch_idx, ch_name in enumerate(channels):
                if ch_name in weather_df.columns:
                    values = weather_df[ch_name].values
                    valid_mask = np.isfinite(values)
                    if valid_mask.any():
                        weather_data[ch_idx, i, j] = np.average(values[valid_mask], weights=weights[valid_mask])
    
    return weather_data

def embed_episode(episode_id, nasa_data, dem_elev, stations, output_dir):
    """Embed one fire episode with 11 channels"""
    
    episode_fires = nasa_data[nasa_data['episode_id'] == episode_id].sort_values('datetime')
    
    if len(episode_fires) < 4:
        return None
    
    timestamps = episode_fires['datetime'].unique()
    if len(timestamps) < 4:
        return None
    
    # Calculate centroid in EPSG:5179
    centroid_lon = episode_fires['LONGITUDE'].mean()
    centroid_lat = episode_fires['LATITUDE'].mean()
    centroid_x, centroid_y = transformer.transform(centroid_lon, centroid_lat)
    
    # Create 30x30 grid centered on fire
    grid_size = 30
    cell_size = 400  # 400m resolution
    half_extent = grid_size * cell_size / 2
    
    x_min = centroid_x - half_extent
    x_max = centroid_x + half_extent
    y_min = centroid_y - half_extent
    y_max = centroid_y + half_extent
    
    x_coords = np.linspace(x_min + cell_size/2, x_max - cell_size/2, grid_size)
    y_coords = np.linspace(y_max - cell_size/2, y_min + cell_size/2, grid_size)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    
    # Extract DEM data for grid
    dem_elev_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    for i in range(grid_size):
        for j in range(grid_size):
            dem_elev_grid[i, j] = get_dem_value(dem_elev, grid_x[i, j], grid_y[i, j])
    
    # Calculate slope from elevation
    dem_slope_grid = calculate_slope(dem_elev_grid).astype(np.float32)
    
    # Process each timestep
    states = []
    ts_list = []
    
    for ts in timestamps:
        ts_fires = episode_fires[episode_fires['datetime'] == ts]
        
        # Create fire state (binary)
        fire_state = np.zeros((grid_size, grid_size), dtype=np.float32)
        
        for _, fire in ts_fires.iterrows():
            fire_x, fire_y = transformer.transform(fire['LONGITUDE'], fire['LATITUDE'])
            
            # Find grid cell
            i = int((y_max - fire_y) / cell_size)
            j = int((fire_x - x_min) / cell_size)
            
            if 0 <= i < grid_size and 0 <= j < grid_size:
                fire_state[i, j] = 1.0
        
        # Load weather data
        ts_dir = Path(f'data/KMA/{ts}')
        weather_df = load_kma_weather(ts_dir, stations)
        weather_grid = interpolate_weather(weather_df, grid_x, grid_y)
        
        # Stack all 11 channels: [2 DEM + 9 Weather]
        features = np.concatenate([
            dem_elev_grid[np.newaxis, :, :],
            dem_slope_grid[np.newaxis, :, :],
            weather_grid
        ], axis=0)  # Shape: (11, 30, 30)
        
        states.append({
            'features': features,
            'fire_state': fire_state,
            'timestamp': ts
        })
        ts_list.append(ts)
    
    # Save episode
    # Combine features (11 channels) + fire_state (1 channel) = 12 channels total
    features_array = np.array([s['features'] for s in states])  # (T, 11, 30, 30)
    fire_states_array = np.array([s['fire_state'] for s in states])  # (T, 30, 30)
    fire_states_array = np.expand_dims(fire_states_array, axis=1)  # (T, 1, 30, 30)
    combined_states = np.concatenate([features_array, fire_states_array], axis=1)  # (T, 12, 30, 30)
    
    output_file = output_dir / f'episode_{episode_id:03d}.npz'
    np.savez_compressed(
        output_file,
        states=combined_states,  # (T, 12, 30, 30) - channels 0-10: env features, channel 11: fire
        timestamps=np.array(ts_list),
        centroid=np.array([centroid_x, centroid_y])
    )
    
    return len(states)

def main():
    print("="*80)
    print("Fire Episode Embedding - 11 Channels (DEM + KMA Weather)")
    print("="*80)
    
    # Load data
    print("\nLoading DEM data...")
    dem_elev = load_dem_data()
    
    print("Loading KMA stations...")
    stations = load_kma_stations()
    
    print("Loading NASA VIIRS fire data...")
    nasa_data = pd.read_parquet('embedded_data/nasa_viirs_embedded.parquet')
    
    output_dir = Path('embedded_data/fire_episodes_11ch')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    episodes = sorted(nasa_data['episode_id'].unique())
    print(f"\nProcessing {len(episodes)} fire episodes...")
    
    success_count = 0
    mel4_count = 0
    
    for idx, ep_id in enumerate(episodes):
        timesteps = embed_episode(ep_id, nasa_data, dem_elev, stations, output_dir)
        
        if timesteps:
            success_count += 1
            if timesteps >= 4:
                mel4_count += 1
        
        if (idx + 1) % 50 == 0:
            print(f"Progress: {idx+1}/{len(episodes)} | Success: {success_count} | MEL>=4: {mel4_count}")
    
    print("="*80)
    print(f"Embedding complete!")
    print(f"Total episodes: {len(episodes)}")
    print(f"Successfully embedded: {success_count}")
    print(f"Episodes with >= 4 timesteps: {mel4_count}")
    print("="*80)

if __name__ == '__main__':
    main()
