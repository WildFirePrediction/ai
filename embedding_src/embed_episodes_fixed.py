"""
Embed fire episodes with CORRECTED coordinate transformations
11 channels: 2 DEM + 9 KMA Weather
Priority: Maximum DEM coverage
"""
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from pyproj import Transformer
import sys

def load_dem():
    """Load DEM and return raster object with correct CRS"""
    dem_file = 'data/DigitalElevationModel/90m_GRS80.tif'
    dem_raster = rasterio.open(dem_file)
    
    # Get actual DEM CRS (not hardcoded EPSG:5179)
    dem_crs = dem_raster.crs
    
    # Create transformer: WGS84 -> DEM CRS
    transformer = Transformer.from_crs("EPSG:4326", dem_crs, always_xy=True)
    
    return dem_raster, transformer

def get_dem_value(raster, x, y):
    """Extract DEM value at coordinate, handle NoData"""
    try:
        row, col = raster.index(x, y)
        if 0 <= row < raster.height and 0 <= col < raster.width:
            val = raster.read(1, window=((row, row+1), (col, col+1)))[0, 0]
            # Check for NoData
            if raster.nodata is not None and val == raster.nodata:
                return np.nan
            if np.isfinite(val) and -100 <= val <= 3000:  # Valid elevation range
                return val
    except:
        pass
    return np.nan

def calculate_slope(elev_grid):
    """Calculate slope from elevation grid"""
    from scipy.ndimage import sobel
    
    # Handle NaN values
    valid_mask = ~np.isnan(elev_grid)
    if valid_mask.sum() < 10:  # Too few valid points
        return np.zeros_like(elev_grid)
    
    # Fill NaN with local mean for gradient calculation
    filled = elev_grid.copy()
    if np.isnan(filled).any():
        filled[np.isnan(filled)] = np.nanmean(elev_grid)
    
    dx = sobel(filled, axis=1, mode='constant')
    dy = sobel(filled, axis=0, mode='constant')
    slope = np.sqrt(dx**2 + dy**2)
    
    # Restore NaN where original was NaN
    slope[~valid_mask] = 0.0  # Use 0 for missing slope
    
    return slope.astype(np.float32)

def load_kma_stations():
    """Load KMA station coordinates"""
    station_file = 'data/KMA/kma_weather_station_5179.csv'
    return pd.read_csv(station_file)

def load_kma_weather(timestamp, stations):
    """Load KMA weather data for timestamp"""
    ts_str = timestamp.strftime('%Y%m%d%H%M')
    kma_dir = Path(f'data/KMA/{ts_str}')
    
    if not kma_dir.exists():
        return None
    
    aws_file = kma_dir / f'AWS_{ts_str}.csv'
    if not aws_file.exists():
        return None
    
    df = pd.read_csv(aws_file)
    
    # Merge with station coordinates
    weather = pd.merge(df, stations[['STN', 'X', 'Y']], on='STN', how='inner')
    
    return weather

def interpolate_weather_idw(weather_df, grid_x, grid_y, channels):
    """IDW interpolation for 9 weather channels"""
    H, W = grid_x.shape
    weather_data = np.zeros((9, H, W), dtype=np.float32)
    
    if weather_df is None or len(weather_df) == 0:
        return weather_data
    
    # For each grid cell
    for i in range(H):
        for j in range(W):
            x, y = grid_x[i, j], grid_y[i, j]
            
            # Calculate distances to all stations
            dx = weather_df['X'].values - x
            dy = weather_df['Y'].values - y
            distances = np.sqrt(dx**2 + dy**2)
            distances = np.maximum(distances, 1.0)  # Avoid division by zero
            
            # IDW weights
            weights = 1.0 / (distances ** 2)
            weights /= weights.sum()
            
            # Interpolate each channel
            for ch_idx, ch_name in enumerate(channels):
                if ch_name in weather_df.columns:
                    values = weather_df[ch_name].values
                    # Handle missing values
                    valid_mask = np.isfinite(values)
                    if valid_mask.any():
                        weather_data[ch_idx, i, j] = np.average(values[valid_mask], weights=weights[valid_mask])
    
    return weather_data

def embed_episode(episode_id, episode_df, dem_raster, dem_transformer, stations, output_dir):
    """Embed one episode with 11 channels"""
    
    # Sort by timestamp
    episode_df = episode_df.sort_values('datetime')
    timestamps = sorted(episode_df['datetime'].unique())
    
    if len(timestamps) < 4:
        return None
    
    # Calculate centroid in WGS84
    centroid_lon = episode_df['LONGITUDE'].mean()
    centroid_lat = episode_df['LATITUDE'].mean()
    
    # Transform to DEM CRS
    centroid_x, centroid_y = dem_transformer.transform(centroid_lon, centroid_lat)
    
    # Create 30x30 grid (400m resolution) in DEM CRS
    grid_size = 30
    cell_size = 400
    half_extent = grid_size * cell_size / 2
    
    x_coords = np.linspace(centroid_x - half_extent + cell_size/2, 
                          centroid_x + half_extent - cell_size/2, grid_size)
    y_coords = np.linspace(centroid_y + half_extent - cell_size/2,
                          centroid_y - half_extent + cell_size/2, grid_size)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    
    # Extract DEM data
    dem_elev = np.zeros((grid_size, grid_size), dtype=np.float32)
    for i in range(grid_size):
        for j in range(grid_size):
            dem_elev[i, j] = get_dem_value(dem_raster, grid_x[i, j], grid_y[i, j])
    
    # Fill NaN with nearest neighbor mean
    if np.isnan(dem_elev).any():
        valid_mask = ~np.isnan(dem_elev)
        if valid_mask.sum() > 0:
            mean_elev = np.nanmean(dem_elev)
            dem_elev[np.isnan(dem_elev)] = mean_elev
        else:
            dem_elev[:] = 100.0  # Default elevation if all NaN
    
    # Calculate slope
    dem_slope = calculate_slope(dem_elev)
    
    # Weather channels
    weather_channels = ['TA', 'HM', 'WS1', 'WD1', 'RN-15m', 'PA', 'PS', 'PS', 'TD']  # 9 channels
    
    # Process each timestep
    states_list = []
    fire_masks_list = []
    ts_list = []
    
    for ts in timestamps:
        ts_fires = episode_df[episode_df['datetime'] == ts]
        
        # Create fire mask
        fire_mask = np.zeros((grid_size, grid_size), dtype=np.float32)
        for _, fire in ts_fires.iterrows():
            fire_x, fire_y = dem_transformer.transform(fire['LONGITUDE'], fire['LATITUDE'])
            i = int((centroid_y + half_extent - fire_y) / cell_size)
            j = int((fire_x - centroid_x + half_extent) / cell_size)
            if 0 <= i < grid_size and 0 <= j < grid_size:
                fire_mask[i, j] = 1.0
        
        # Load weather
        weather_df = load_kma_weather(ts, stations)
        weather_grid = interpolate_weather_idw(weather_df, grid_x, grid_y, weather_channels)
        
        # Stack: 2 DEM + 9 weather = 11 channels
        features = np.concatenate([
            dem_elev[np.newaxis, :, :],
            dem_slope[np.newaxis, :, :],
            weather_grid
        ], axis=0)  # (11, 30, 30)
        
        states_list.append(features)
        fire_masks_list.append(fire_mask)
        ts_list.append(ts.strftime('%Y-%m-%d %H:%M:%S'))  # String format to avoid pickle
    
    # Save episode
    states = np.array(states_list, dtype=np.float32)  # (T, 11, 30, 30)
    fire_masks = np.array(fire_masks_list, dtype=np.float32)  # (T, 30, 30)
    
    output_file = output_dir / f'episode_{episode_id:04d}.npz'
    np.savez_compressed(
        output_file,
        states=states,
        fire_masks=fire_masks,
        timestamps=np.array(ts_list),  # String array, no pickle needed
        centroid=np.array([centroid_x, centroid_y], dtype=np.float32)
    )
    
    return len(timestamps)

def main():
    print("="*80)
    print("EMBEDDING FIRE EPISODES - FIXED VERSION")
    print("11 Channels: 2 DEM + 9 Weather")
    print("="*80)
    
    # Load DEM with correct CRS
    print("\nLoading DEM with correct CRS...")
    dem_raster, dem_transformer = load_dem()
    print(f"  DEM CRS: {dem_raster.crs}")
    print(f"  DEM shape: {dem_raster.shape}")
    
    # Load KMA stations
    print("\nLoading KMA stations...")
    stations = load_kma_stations()
    print(f"  Stations: {len(stations)}")
    
    # Load reclustered episodes
    print("\nLoading reclustered episodes...")
    nasa_df = pd.read_parquet('embedded_data/nasa_viirs_reclustered.parquet')
    print(f"  Total detections: {len(nasa_df):,}")
    
    episodes = nasa_df['episode_id'].unique()
    print(f"  Episodes: {len(episodes)}")
    
    # Create output directory
    output_dir = Path('embedded_data/fire_episodes_11ch_fixed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Embed each episode
    print(f"\nEmbedding episodes...")
    success_count = 0
    mel4_count = 0
    
    for idx, ep_id in enumerate(episodes):
        ep_data = nasa_df[nasa_df['episode_id'] == ep_id]
        
        timesteps = embed_episode(ep_id, ep_data, dem_raster, dem_transformer, stations, output_dir)
        
        if timesteps:
            success_count += 1
            if timesteps >= 4:
                mel4_count += 1
        
        if (idx + 1) % 100 == 0 or (idx + 1) == len(episodes):
            print(f"  Progress: {idx+1}/{len(episodes)} | Success: {success_count} | MEL>=4: {mel4_count}")
            sys.stdout.flush()
    
    dem_raster.close()
    
    print("\n" + "="*80)
    print("EMBEDDING COMPLETE")
    print(f"Total episodes: {len(episodes)}")
    print(f"Successfully embedded: {success_count}")
    print(f"Episodes with >= 4 timesteps: {mel4_count}")
    print(f"Output: {output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
