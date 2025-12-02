"""
Embed fire episodes with 10 channels:
- 2 DEM (slope, aspect)
- 3 Wind (wind_speed, wind_u, wind_v)
- 1 NDVI
- 4 FSM (one-hot encoded)

Priority: Clean wind-driven fire spread prediction
"""
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from pyproj import Transformer
import sys

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

def extract_grid_features(grid_x, grid_y, dem_raster, ndvi_raster, fsm_raster):
    """Extract DEM (slope, aspect), NDVI, FSM for grid"""
    H, W = grid_x.shape

    # Initialize
    slope = np.zeros((H, W), dtype=np.float32)
    aspect = np.zeros((H, W), dtype=np.float32)
    ndvi = np.zeros((H, W), dtype=np.float32)
    fsm_raw = np.zeros((H, W), dtype=np.int32)

    # Sample each cell
    for i in range(H):
        for j in range(W):
            x, y = grid_x[i, j], grid_y[i, j]

            # DEM: bands 2=slope, 3=aspect
            slope_val = sample_raster(dem_raster, x, y, band=2)
            aspect_val = sample_raster(dem_raster, x, y, band=3)
            slope[i, j] = slope_val if np.isfinite(slope_val) else 0.0
            aspect[i, j] = aspect_val if np.isfinite(aspect_val) else 0.0

            # NDVI
            ndvi_val = sample_raster(ndvi_raster, x, y, band=1)
            ndvi[i, j] = ndvi_val if np.isfinite(ndvi_val) else 0.5  # Default moderate vegetation

            # FSM (categorical)
            fsm_val = sample_raster(fsm_raster, x, y, band=1)
            fsm_raw[i, j] = int(fsm_val) if np.isfinite(fsm_val) else 0

    # One-hot encode FSM (5 classes: 0,1,2,3,4 -> 4 channels, class 0 is background)
    # Use classes 1-4 for one-hot encoding
    fsm_onehot = np.zeros((4, H, W), dtype=np.float32)
    for class_id in range(1, 5):
        fsm_onehot[class_id-1] = (fsm_raw == class_id).astype(np.float32)

    return slope, aspect, ndvi, fsm_onehot

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

    try:
        df = pd.read_csv(aws_file)

        # Merge with station coordinates
        weather = pd.merge(df, stations[['STN', 'X', 'Y']], on='STN', how='inner')

        # Filter out missing wind data (-99.x values)
        weather = weather[
            (weather['WS1'] >= 0) & (weather['WS1'] < 100) &
            (weather['WD1'] >= 0) & (weather['WD1'] <= 360)
        ]

        return weather
    except Exception as e:
        print(f"    Error loading {aws_file}: {e}")
        return None

def interpolate_wind_idw(weather_df, grid_x, grid_y):
    """
    IDW interpolation for wind: wind_speed, wind_u, wind_v

    Meteorological convention:
    - WD: direction FROM which wind blows (0=North, 90=East, 180=South, 270=West)
    - u-component (eastward wind): -WS * sin(WD)
    - v-component (northward wind): -WS * cos(WD)
    """
    H, W = grid_x.shape
    wind_data = np.zeros((3, H, W), dtype=np.float32)

    if weather_df is None or len(weather_df) == 0:
        return wind_data  # Return zeros if no data

    # Precompute wind components at stations
    wind_speed = weather_df['WS1'].values
    wind_dir_deg = weather_df['WD1'].values
    wind_dir_rad = np.deg2rad(wind_dir_deg)

    # Wind components (meteorological convention)
    wind_u_stations = -wind_speed * np.sin(wind_dir_rad)
    wind_v_stations = -wind_speed * np.cos(wind_dir_rad)

    station_x = weather_df['X'].values
    station_y = weather_df['Y'].values

    # For each grid cell
    for i in range(H):
        for j in range(W):
            x, y = grid_x[i, j], grid_y[i, j]

            # Calculate distances to all stations
            dx = station_x - x
            dy = station_y - y
            distances = np.sqrt(dx**2 + dy**2)
            distances = np.maximum(distances, 1.0)  # Avoid division by zero

            # IDW weights (power=2)
            weights = 1.0 / (distances ** 2)
            weights /= weights.sum()

            # Interpolate wind components
            wind_data[0, i, j] = np.average(wind_speed, weights=weights)
            wind_data[1, i, j] = np.average(wind_u_stations, weights=weights)
            wind_data[2, i, j] = np.average(wind_v_stations, weights=weights)

    return wind_data

def embed_episode(episode_id, episode_df, dem_raster, ndvi_raster, fsm_raster,
                 transformer, stations, output_dir):
    """Embed one episode with 10 channels"""

    # Sort by timestamp
    episode_df = episode_df.sort_values('datetime')
    timestamps = sorted(episode_df['datetime'].unique())

    if len(timestamps) < 4:
        return None, "Too few timesteps"

    # Calculate centroid in WGS84
    centroid_lon = episode_df['LONGITUDE'].mean()
    centroid_lat = episode_df['LATITUDE'].mean()

    # Transform to target CRS
    centroid_x, centroid_y = transformer.transform(centroid_lon, centroid_lat)

    # Create 30x30 grid (400m resolution)
    grid_size = 30
    cell_size = 400
    half_extent = grid_size * cell_size / 2

    x_coords = np.linspace(centroid_x - half_extent + cell_size/2,
                          centroid_x + half_extent - cell_size/2, grid_size)
    y_coords = np.linspace(centroid_y + half_extent - cell_size/2,
                          centroid_y - half_extent + cell_size/2, grid_size)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    # Extract static spatial features (DEM, NDVI, FSM)
    slope, aspect, ndvi, fsm_onehot = extract_grid_features(
        grid_x, grid_y, dem_raster, ndvi_raster, fsm_raster
    )

    # Process each timestep
    states_list = []
    fire_masks_list = []
    ts_list = []
    missing_weather_count = 0

    for ts in timestamps:
        ts_fires = episode_df[episode_df['datetime'] == ts]

        # Create fire mask
        fire_mask = np.zeros((grid_size, grid_size), dtype=np.float32)
        for _, fire in ts_fires.iterrows():
            fire_x, fire_y = transformer.transform(fire['LONGITUDE'], fire['LATITUDE'])
            i = int((centroid_y + half_extent - fire_y) / cell_size)
            j = int((fire_x - centroid_x + half_extent) / cell_size)
            if 0 <= i < grid_size and 0 <= j < grid_size:
                fire_mask[i, j] = 1.0

        # Load and interpolate wind data
        weather_df = load_kma_weather(ts, stations)
        if weather_df is None or len(weather_df) == 0:
            missing_weather_count += 1

        wind_grid = interpolate_wind_idw(weather_df, grid_x, grid_y)

        # Stack: 2 DEM + 3 Wind + 1 NDVI + 4 FSM = 10 channels
        features = np.concatenate([
            slope[np.newaxis, :, :],      # Ch 0
            aspect[np.newaxis, :, :],     # Ch 1
            wind_grid,                     # Ch 2-4: wind_speed, wind_u, wind_v
            ndvi[np.newaxis, :, :],       # Ch 5
            fsm_onehot                     # Ch 6-9: FSM classes 1-4
        ], axis=0)  # (10, 30, 30)

        states_list.append(features)
        fire_masks_list.append(fire_mask)
        ts_list.append(ts.strftime('%Y-%m-%d %H:%M:%S'))

    # Check weather coverage
    weather_coverage = 1.0 - (missing_weather_count / len(timestamps))
    if weather_coverage < 0.5:
        return None, f"Low weather coverage: {weather_coverage:.1%}"

    # Save episode
    states = np.array(states_list, dtype=np.float32)  # (T, 10, 30, 30)
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
    print("EMBEDDING FIRE EPISODES - 10 CHANNELS (WIND-FOCUSED)")
    print("="*80)
    print("Channels:")
    print("  0-1:  DEM (slope, aspect)")
    print("  2-4:  Wind (speed, u-component, v-component)")
    print("  5:    NDVI (vegetation)")
    print("  6-9:  FSM (forest susceptibility, one-hot)")
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

    episodes = nasa_df['episode_id'].unique()
    print(f"  Episodes: {len(episodes)}")

    # Create output directory
    output_dir = Path('embedded_data/fire_episodes_10ch_wind')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Embed each episode
    print(f"\nEmbedding episodes...")
    success_count = 0
    mel4_count = 0
    total_weather_coverage = []
    skipped_reasons = {}

    for idx, ep_id in enumerate(episodes):
        ep_data = nasa_df[nasa_df['episode_id'] == ep_id]

        result = embed_episode(
            ep_id, ep_data, dem_raster, ndvi_raster, fsm_raster,
            transformer, stations, output_dir
        )

        if result[0] is not None:
            timesteps, weather_cov = result
            success_count += 1
            total_weather_coverage.append(weather_cov)
            if timesteps >= 4:
                mel4_count += 1
        else:
            reason = result[1]
            skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

        if (idx + 1) % 100 == 0 or (idx + 1) == len(episodes):
            print(f"  Progress: {idx+1}/{len(episodes)} | "
                  f"Success: {success_count} | MEL>=4: {mel4_count}")
            sys.stdout.flush()

    # Close rasters
    dem_raster.close()
    ndvi_raster.close()
    fsm_raster.close()

    # Summary
    print("\n" + "="*80)
    print("EMBEDDING COMPLETE")
    print(f"Total episodes: {len(episodes)}")
    print(f"Successfully embedded: {success_count}")
    print(f"Episodes with >= 4 timesteps: {mel4_count}")
    if total_weather_coverage:
        print(f"Average weather coverage: {np.mean(total_weather_coverage):.1%}")
        print(f"Median weather coverage: {np.median(total_weather_coverage):.1%}")
    print(f"\nSkipped episodes:")
    for reason, count in skipped_reasons.items():
        print(f"  {reason}: {count}")
    print(f"Output: {output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()
