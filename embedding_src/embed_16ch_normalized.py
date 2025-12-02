"""
Embed fire episodes with 16 channels - NORMALIZED FEATURES:
- 2 DEM (slope, aspect) - already normalized
- 9 Weather (temp, humidity, wind_speed, wind_dir, precip, pressure, cloud, visibility, dew_point) - NORMALIZED
- 1 NDVI (vegetation) - already normalized
- 4 FSM (forest susceptibility, one-hot) - already normalized

All features normalized to similar scales for proper gradient flow
"""
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from pyproj import Transformer
import sys

# Normalization parameters for weather features (based on Korean climate)
WEATHER_NORM = {
    'temperature': {'min': -30, 'max': 40},  # Celsius
    'humidity': {'min': 0, 'max': 100},  # Percentage
    'wind_speed': {'min': 0, 'max': 30},  # m/s
    'wind_direction': {'min': 0, 'max': 360},  # Degrees
    'precipitation': {'min': 0, 'max': 50},  # mm
    'pressure': {'min': 950, 'max': 1050},  # hPa
    'cloud_cover': {'min': 0, 'max': 10},  # Oktas (0-10)
    'visibility': {'min': 0, 'max': 40000},  # meters
    'dew_point': {'min': -40, 'max': 30}  # Celsius
}

def normalize_weather(weather_data):
    """
    Normalize weather data to 0-1 range using min-max scaling

    Args:
        weather_data: (9, H, W) array with weather channels

    Returns:
        Normalized weather data (9, H, W)
    """
    channels = ['temperature', 'humidity', 'wind_speed', 'wind_direction',
                'precipitation', 'pressure', 'cloud_cover', 'visibility', 'dew_point']

    normalized = weather_data.copy()

    for ch_idx, ch_name in enumerate(channels):
        min_val = WEATHER_NORM[ch_name]['min']
        max_val = WEATHER_NORM[ch_name]['max']

        # Min-max normalization: (x - min) / (max - min)
        normalized[ch_idx] = (weather_data[ch_idx] - min_val) / (max_val - min_val)

        # Clip to [0, 1] in case of outliers
        normalized[ch_idx] = np.clip(normalized[ch_idx], 0.0, 1.0)

    return normalized

def load_spatial_data():
    """Load all spatial rasters (DEM, NDVI, FSM)"""
    # DEM (bands: elevation, slope, aspect)
    dem_file = 'embedded_data/dem_slope_aspect_FINAL.tif'
    dem_raster = rasterio.open(dem_file)

    # NDVI
    ndvi_file = 'embedded_data/ndvi_FINAL.tif'
    ndvi_raster = rasterio.open(ndvi_file)

    # FSM
    fsm_file = 'embedded_data/fsm_FINAL.tif'
    fsm_raster = rasterio.open(fsm_file)

    # All should have same CRS
    print(f"  DEM CRS: {dem_raster.crs}")
    print(f"  NDVI CRS: {ndvi_raster.crs}")
    print(f"  FSM CRS: {fsm_raster.crs}")

    # Create transformer: WGS84 -> DEM CRS
    transformer = Transformer.from_crs("EPSG:4326", dem_raster.crs, always_xy=True)

    return dem_raster, ndvi_raster, fsm_raster, transformer

def sample_raster(raster, x, y, band=1):
    """Sample value from raster at coordinate"""
    try:
        row, col = raster.index(x, y)
        if 0 <= row < raster.height and 0 <= col < raster.width:
            val = raster.read(band, window=((row, row+1), (col, col+1)))[0, 0]
            if raster.nodata is not None and val == raster.nodata:
                return np.nan
            if np.isfinite(val):
                return val
    except:
        pass
    return np.nan

def extract_spatial_features(grid_x, grid_y, dem_raster, ndvi_raster, fsm_raster):
    """Extract DEM, NDVI, FSM for 30x30 grid"""
    H, W = grid_x.shape

    # DEM: slope (band 2), aspect (band 3)
    slope = np.zeros((H, W), dtype=np.float32)
    aspect = np.zeros((H, W), dtype=np.float32)

    # NDVI
    ndvi = np.zeros((H, W), dtype=np.float32)

    # FSM (values 1-4)
    fsm = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            x, y = grid_x[i, j], grid_y[i, j]

            # DEM
            slope_val = sample_raster(dem_raster, x, y, band=2)
            aspect_val = sample_raster(dem_raster, x, y, band=3)
            slope[i, j] = slope_val if np.isfinite(slope_val) else 0.0
            aspect[i, j] = aspect_val if np.isfinite(aspect_val) else 0.0

            # NDVI
            ndvi_val = sample_raster(ndvi_raster, x, y, band=1)
            ndvi[i, j] = ndvi_val if np.isfinite(ndvi_val) else 0.0

            # FSM
            fsm_val = sample_raster(fsm_raster, x, y, band=1)
            fsm[i, j] = fsm_val if np.isfinite(fsm_val) else 0.0

    # One-hot encode FSM (4 classes: 1-4)
    fsm_onehot = np.zeros((4, H, W), dtype=np.float32)
    for class_idx in range(1, 5):
        fsm_onehot[class_idx-1] = (fsm == class_idx).astype(np.float32)

    return slope, aspect, ndvi, fsm_onehot

def load_kma_stations():
    """Load KMA station coordinates"""
    stations = pd.read_csv('data/KMA/kma_weather_station_5179.csv')
    return stations

def load_kma_weather(timestamp, stations):
    """
    Load KMA weather for a specific timestamp
    Returns dataframe with columns: STN, X, Y, and weather fields
    """
    ts_str = timestamp.strftime('%Y%m%d%H%M')
    kma_dir = Path(f'data/KMA/{ts_str}')

    if not kma_dir.exists():
        return None

    aws_file = kma_dir / f'AWS_{ts_str}.csv'
    if not aws_file.exists():
        return None

    try:
        df = pd.read_csv(aws_file)

        # Merge with station coordinates
        weather = pd.merge(df, stations[['STN', 'X', 'Y']], on='STN', how='inner')

        # Map columns to expected names
        weather['temperature'] = weather['TA'] if 'TA' in weather.columns else 0
        weather['humidity'] = weather['HM'] if 'HM' in weather.columns else 0
        weather['wind_speed'] = weather['WS1'] if 'WS1' in weather.columns else 0
        weather['wind_direction'] = weather['WD1'] if 'WD1' in weather.columns else 0
        weather['precipitation'] = weather['RN-15m'] if 'RN-15m' in weather.columns else 0
        weather['pressure'] = weather['PA'] if 'PA' in weather.columns else 0
        weather['dew_point'] = weather['TD'] if 'TD' in weather.columns else 0
        weather['cloud_cover'] = 0  # Not available
        weather['visibility'] = 0  # Not available

        weather['x'] = weather['X']
        weather['y'] = weather['Y']

        return weather
    except Exception as e:
        print(f"Error loading weather for {ts_str}: {e}")
        return None

def interpolate_weather_idw(weather_df, grid_x, grid_y):
    """
    IDW interpolation for 9 weather channels (NOT NORMALIZED YET)
    Normalization happens later in embed_episode()
    """
    channels = ['temperature', 'humidity', 'wind_speed', 'wind_direction',
                'precipitation', 'pressure', 'cloud_cover', 'visibility', 'dew_point']

    H, W = grid_x.shape
    weather_data = np.zeros((9, H, W), dtype=np.float32)

    if weather_df is None or len(weather_df) == 0:
        # Return zeros if no weather data (will be normalized to 0.5 range)
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

def create_grid(centroid_x, centroid_y, grid_size=30, cell_size=400):
    """Create 30x30 grid centered on fire centroid"""
    half_extent = (grid_size * cell_size) / 2  # 6000m = 6km radius

    x_min = centroid_x - half_extent
    x_max = centroid_x + half_extent
    y_min = centroid_y - half_extent
    y_max = centroid_y + half_extent

    x_coords = np.linspace(x_min, x_max, grid_size)
    y_coords = np.linspace(y_max, y_min, grid_size)  # Reverse Y for raster convention

    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    return grid_x, grid_y

def embed_episode(episode_id, episode_df, dem_raster, ndvi_raster, fsm_raster,
                  stations, transformer, output_dir):
    """
    Embed single fire episode with 16 NORMALIZED channels

    Returns:
        (num_timesteps, weather_coverage) or (None, error_message)
    """
    # Sort by timestamp
    episode_df = episode_df.sort_values('datetime')
    timestamps = sorted(episode_df['datetime'].unique())

    if len(timestamps) < 4:
        return None, "Too few timesteps"

    # Calculate centroid in WGS84
    centroid_lon = episode_df['LONGITUDE'].mean()
    centroid_lat = episode_df['LATITUDE'].mean()

    # Transform to projected CRS
    centroid_x, centroid_y = transformer.transform(centroid_lon, centroid_lat)

    # Create 30x30 grid
    grid_x, grid_y = create_grid(centroid_x, centroid_y)

    # Extract static spatial features (DEM, NDVI, FSM) - already normalized
    slope, aspect, ndvi, fsm_onehot = extract_spatial_features(
        grid_x, grid_y, dem_raster, ndvi_raster, fsm_raster
    )

    # Process each timestep
    states_list = []
    fire_masks_list = []
    ts_list = []
    missing_weather_count = 0

    for ts in timestamps:
        ts_df = episode_df[episode_df['datetime'] == ts]

        # Create fire mask
        fire_mask = np.zeros((30, 30), dtype=np.float32)
        for _, row in ts_df.iterrows():
            px, py = transformer.transform(row['LONGITUDE'], row['LATITUDE'])

            # Find grid cell
            col = np.argmin(np.abs(grid_x[0, :] - px))
            row_idx = np.argmin(np.abs(grid_y[:, 0] - py))

            if 0 <= row_idx < 30 and 0 <= col < 30:
                fire_mask[row_idx, col] = 1.0

        # Load weather for this timestep
        weather_df = load_kma_weather(ts, stations)

        if weather_df is None or len(weather_df) == 0:
            missing_weather_count += 1

        # Interpolate weather (raw values)
        weather_grid_raw = interpolate_weather_idw(weather_df, grid_x, grid_y)

        # NORMALIZE weather to 0-1 range
        weather_grid = normalize_weather(weather_grid_raw)

        # Stack: 2 DEM + 9 Weather + 1 NDVI + 4 FSM = 16 channels (ALL NORMALIZED)
        features = np.concatenate([
            slope[np.newaxis, :, :],      # Ch 0: slope (0-1)
            aspect[np.newaxis, :, :],     # Ch 1: aspect (0-1)
            weather_grid,                  # Ch 2-10: weather (0-1)
            ndvi[np.newaxis, :, :],       # Ch 11: NDVI (0-1)
            fsm_onehot                     # Ch 12-15: FSM (0-1)
        ], axis=0)  # (16, 30, 30)

        states_list.append(features)
        fire_masks_list.append(fire_mask)
        ts_list.append(ts.strftime('%Y-%m-%d %H:%M:%S'))

    # Check weather coverage
    weather_coverage = 1.0 - (missing_weather_count / len(timestamps))
    if weather_coverage < 0.5:
        return None, f"Low weather coverage: {weather_coverage:.1%}"

    # Save episode
    states = np.array(states_list, dtype=np.float32)  # (T, 16, 30, 30)
    fire_masks = np.array(fire_masks_list, dtype=np.float32)  # (T, 30, 30)

    output_file = output_dir / f'episode_{episode_id:04d}.npz'
    np.savez_compressed(
        output_file,
        states=states,
        fire_masks=fire_masks,
        timestamps=np.array(ts_list),
        centroid=np.array([centroid_x, centroid_y], dtype=np.float32),
        weather_coverage=weather_coverage
    )

    return len(timestamps), weather_coverage

def main():
    print("="*80)
    print("EMBEDDING FIRE EPISODES - 16 CHANNELS (NORMALIZED)")
    print("="*80)
    print("Channels (ALL NORMALIZED TO 0-1 RANGE):")
    print("  0-1:   DEM (slope, aspect)")
    print("  2-10:  Weather (temp, humidity, wind_speed, wind_dir, precip,")
    print("                  pressure, cloud, visibility, dew_point)")
    print("  11:    NDVI (vegetation)")
    print("  12-15: FSM (forest susceptibility, one-hot)")
    print("="*80)

    # Load spatial data
    print("\nLoading spatial data...")
    dem_raster, ndvi_raster, fsm_raster, transformer = load_spatial_data()

    # Load KMA stations
    print("\nLoading KMA stations...")
    stations = load_kma_stations()
    print(f"  Stations: {len(stations)}")

    # Load reclustered episodes
    print("\nLoading reclustered episodes...")
    nasa_df = pd.read_parquet('embedded_data/nasa_viirs_reclustered.parquet')
    print(f"  Total detections: {len(nasa_df):,}")

    episode_ids = nasa_df['episode_id'].unique()
    print(f"  Episodes: {len(episode_ids)}")

    # Create output directory
    output_dir = Path('embedded_data/fire_episodes_16ch_normalized')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Embed episodes
    print("\nEmbedding episodes...")
    success_count = 0
    mel4_count = 0
    error_count = 0

    for idx, episode_id in enumerate(episode_ids):
        # Skip if already exists
        output_file = output_dir / f'episode_{episode_id:04d}.npz'
        if output_file.exists():
            success_count += 1
            # Check if MEL >= 4
            try:
                data = np.load(output_file)
                if len(data['states']) >= 5:
                    mel4_count += 1
            except:
                pass
            continue

        try:
            # Get episode data
            episode_df = nasa_df[nasa_df['episode_id'] == episode_id].copy()

            result, info = embed_episode(
                episode_id, episode_df, dem_raster, ndvi_raster, fsm_raster,
                stations, transformer, output_dir
            )

            if result is not None:
                success_count += 1
                num_timesteps, weather_cov = result, info

                if num_timesteps >= 5:  # MEL >= 4
                    mel4_count += 1
            else:
                error_count += 1
        except Exception as e:
            error_count += 1
            print(f"  ERROR on episode {episode_id}: {e}")
            continue

        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx+1}/{len(episode_ids)} | "
                  f"Success: {success_count} | MEL>=4: {mel4_count} | Errors: {error_count}")

    print("\n" + "="*80)
    print("EMBEDDING COMPLETE")
    print("="*80)
    print(f"Total episodes: {len(episode_ids)}")
    print(f"Successfully embedded: {success_count}")
    print(f"Episodes with >= 4 timesteps: {mel4_count}")
    print(f"Errors/skipped: {error_count}")
    print(f"Output directory: {output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
