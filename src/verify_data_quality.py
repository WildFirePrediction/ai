"""
Data Quality Verification Script
Checks:
1. Lat/lon consistency across all data sources
2. Time consistency across all data sources
3. Temporal alignment between KMA AWS weather data and NASA fire data
4. Verification of embedded data quality
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DATA QUALITY VERIFICATION")
print("=" * 80)

# Paths
root_dir = Path('/')
data_dir = Path('../data')
tilling_dir = Path('../tilling_data')
filtered_fires_dir = data_dir / 'filtered_fires'

# =============================================================================
# 1. CHECK FILTERED FIRE DATA
# =============================================================================
print("\n" + "=" * 80)
print("1. FILTERED FIRE DATA VERIFICATION")
print("=" * 80)

if (filtered_fires_dir / 'filtered_viirs.parquet').exists():
    print("\n[✓] Loading filtered VIIRS data...")
    df_fire = pd.read_parquet(filtered_fires_dir / 'filtered_viirs.parquet')
    print(f"  Total fire detections: {len(df_fire):,}")
    print(f"\n  Columns: {df_fire.columns.tolist()}")

    # Check lat/lon columns
    print("\n  Checking latitude/longitude columns...")
    lat_col = 'LATITUDE' if 'LATITUDE' in df_fire.columns else 'latitude'
    lon_col = 'LONGITUDE' if 'LONGITUDE' in df_fire.columns else 'longitude'

    if lat_col in df_fire.columns and lon_col in df_fire.columns:
        print(f"    [✓] Found lat/lon columns: {lat_col}, {lon_col}")
        print(f"    Latitude range: [{df_fire[lat_col].min():.4f}, {df_fire[lat_col].max():.4f}]")
        print(f"    Longitude range: [{df_fire[lon_col].min():.4f}, {df_fire[lon_col].max():.4f}]")

        # Check for NaN values
        lat_nan = df_fire[lat_col].isna().sum()
        lon_nan = df_fire[lon_col].isna().sum()
        if lat_nan > 0 or lon_nan > 0:
            print(f"    [⚠️] WARNING: Found {lat_nan} NaN latitudes and {lon_nan} NaN longitudes")
        else:
            print(f"    [✓] No NaN values in lat/lon")
    else:
        print(f"    [✗] ERROR: Lat/lon columns not found!")

    # Check projected coordinates (x, y)
    print("\n  Checking projected coordinates (x, y)...")
    if 'x' in df_fire.columns and 'y' in df_fire.columns:
        print(f"    [✓] Found projected coordinates: x, y")
        print(f"    X range: [{df_fire['x'].min():.2f}, {df_fire['x'].max():.2f}] meters")
        print(f"    Y range: [{df_fire['y'].min():.2f}, {df_fire['y'].max():.2f}] meters")

        x_nan = df_fire['x'].isna().sum()
        y_nan = df_fire['y'].isna().sum()
        if x_nan > 0 or y_nan > 0:
            print(f"    [⚠️] WARNING: Found {x_nan} NaN x and {y_nan} NaN y coordinates")
        else:
            print(f"    [✓] No NaN values in x/y")
    else:
        print(f"    [✗] ERROR: Projected coordinates (x, y) not found!")

    # Check datetime
    print("\n  Checking datetime consistency...")
    if 'datetime' in df_fire.columns:
        print(f"    [✓] Found datetime column")
        print(f"    Date range: {df_fire['datetime'].min()} to {df_fire['datetime'].max()}")

        dt_nan = df_fire['datetime'].isna().sum()
        if dt_nan > 0:
            print(f"    [⚠️] WARNING: Found {dt_nan} NaN datetime values")
        else:
            print(f"    [✓] No NaN values in datetime")
    elif 'acq_date' in df_fire.columns and 'acq_time' in df_fire.columns:
        print(f"    [⚠️] WARNING: datetime column not found, but acq_date and acq_time exist")
        print(f"    Creating datetime from acq_date and acq_time...")
        df_fire['datetime'] = pd.to_datetime(
            df_fire['acq_date'] + ' ' + df_fire['acq_time'].astype(str).str.zfill(4),
            format='%Y-%m-%d %H%M'
        )
        print(f"    Date range: {df_fire['datetime'].min()} to {df_fire['datetime'].max()}")
    else:
        print(f"    [✗] ERROR: No datetime information found!")

    # Check weather data columns
    print("\n  Checking KMA weather data columns...")
    weather_cols = ['w', 'd_x', 'd_y', 'rh', 'r', 'te']  # wind speed, wind direction x/y, humidity, rain, temp
    found_weather = []
    missing_weather = []

    for col in weather_cols:
        if col in df_fire.columns:
            found_weather.append(col)
        else:
            missing_weather.append(col)

    if found_weather:
        print(f"    [✓] Found weather columns: {found_weather}")
        for col in found_weather:
            nan_count = df_fire[col].isna().sum()
            nan_pct = 100 * nan_count / len(df_fire)
            print(f"      - {col}: {nan_count:,} NaN values ({nan_pct:.2f}%)")
            if nan_count == 0:
                print(f"        [✓] No NaN values in {col}")
            elif nan_pct > 50:
                print(f"        [⚠️] WARNING: >50% NaN values in {col}")

            # Show statistics for non-NaN values
            if nan_count < len(df_fire):
                valid_data = df_fire[col].dropna()
                print(f"        Range: [{valid_data.min():.2f}, {valid_data.max():.2f}], Mean: {valid_data.mean():.2f}")

    if missing_weather:
        print(f"    [⚠️] WARNING: Missing weather columns: {missing_weather}")

    # Check fire intensity columns
    print("\n  Checking fire intensity/temperature columns...")
    fire_cols = ['i', 'te', 'FRP', 'BRIGHTNESS', 'BRIGHT_T31']
    for col in fire_cols:
        if col in df_fire.columns:
            nan_count = df_fire[col].isna().sum()
            valid_data = df_fire[col].dropna()
            print(f"    [✓] Found {col}: {nan_count} NaN, Range: [{valid_data.min():.2f}, {valid_data.max():.2f}]")

else:
    print("[✗] ERROR: Filtered VIIRS data not found!")
    df_fire = None

# =============================================================================
# 2. CHECK TILLING DATA STRUCTURE
# =============================================================================
print("\n" + "=" * 80)
print("2. TILLING DATA VERIFICATION")
print("=" * 80)

if tilling_dir.exists():
    print("\n[✓] Tilling data directory found")

    # Check window regions
    if (tilling_dir / 'window_regions.parquet').exists():
        print("\n  [✓] Loading window regions...")
        df_regions = pd.read_parquet(tilling_dir / 'window_regions.parquet')
        print(f"    Total windows: {len(df_regions)}")
        print(f"    Columns: {df_regions.columns.tolist()}")

        # Check region metadata
        print(f"\n    Region size statistics:")
        print(f"      Height: mean={df_regions['height'].mean():.1f}, range=[{df_regions['height'].min()}, {df_regions['height'].max()}]")
        print(f"      Width: mean={df_regions['width'].mean():.1f}, range=[{df_regions['width'].min()}, {df_regions['width'].max()}]")
        print(f"      Detections per window: mean={df_regions['n_detections'].mean():.1f}, range=[{df_regions['n_detections'].min()}, {df_regions['n_detections'].max()}]")
        print(f"      Duration (hours): mean={df_regions['duration_hours'].mean():.1f}, range=[{df_regions['duration_hours'].min():.1f}, {df_regions['duration_hours'].max():.1f}]")

        # Check a sample region file
        regions_dir = tilling_dir / 'regions'
        if regions_dir.exists():
            region_files = list(regions_dir.glob('window_region_*.npz'))
            if region_files:
                print(f"\n    [✓] Found {len(region_files)} region files")

                # Load first region as sample
                sample_region = np.load(region_files[0], allow_pickle=True)
                print(f"\n    Sample region ({region_files[0].name}):")
                print(f"      Keys: {list(sample_region.keys())}")

                if 'continuous_features' in sample_region:
                    cont_shape = sample_region['continuous_features'].shape
                    print(f"      Continuous features shape: {cont_shape}")
                    print(f"      Feature names: {sample_region.get('feature_names', 'N/A')}")

                    # Check for NaN values
                    cont_data = sample_region['continuous_features']
                    nan_count = np.isnan(cont_data).sum()
                    total_vals = np.prod(cont_data.shape)
                    if nan_count > 0:
                        print(f"      [⚠️] WARNING: {nan_count}/{total_vals} NaN values in continuous features ({100*nan_count/total_vals:.2f}%)")
                    else:
                        print(f"      [✓] No NaN values in continuous features")

                if 'world_bounds' in sample_region:
                    wb = sample_region['world_bounds'].item()
                    print(f"      World bounds: x=[{wb['x_min']:.2f}, {wb['x_max']:.2f}], y=[{wb['y_min']:.2f}, {wb['y_max']:.2f}]")
            else:
                print(f"    [✗] ERROR: No region files found!")
        else:
            print(f"    [✗] ERROR: Regions directory not found!")
    else:
        print("  [✗] ERROR: window_regions.parquet not found!")

    # Check temporal sequences
    sequences_dir = tilling_dir / 'sequences'
    if sequences_dir.exists():
        seq_files = list(sequences_dir.glob('window_*.npz'))
        if seq_files:
            print(f"\n  [✓] Found {len(seq_files)} temporal sequence files")

            # Load first sequence as sample
            sample_seq = np.load(seq_files[0], allow_pickle=True)
            print(f"\n    Sample sequence ({seq_files[0].name}):")
            print(f"      Keys: {list(sample_seq.keys())}")

            if 'fire_masks' in sample_seq:
                fm_shape = sample_seq['fire_masks'].shape
                print(f"      Fire masks shape: {fm_shape} (T, H, W)")
                print(f"      Number of timesteps: {fm_shape[0]}")

            if 'weather_states' in sample_seq:
                ws_shape = sample_seq['weather_states'].shape
                print(f"      Weather states shape: {ws_shape} (T, 5)")

                # Check for NaN values in weather
                weather_data = sample_seq['weather_states']
                nan_count = np.isnan(weather_data).sum()
                if nan_count > 0:
                    print(f"      [⚠️] WARNING: {nan_count} NaN values in weather states")
                else:
                    print(f"      [✓] No NaN values in weather states")

            if 'timesteps' in sample_seq:
                timesteps = sample_seq['timesteps']
                print(f"      Timesteps: {len(timesteps)} values")
                # Convert to datetime for display
                ts_datetimes = pd.to_datetime(timesteps)
                print(f"      Time range: {ts_datetimes.min()} to {ts_datetimes.max()}")
        else:
            print(f"  [✗] ERROR: No sequence files found!")
    else:
        print(f"  [✗] ERROR: Sequences directory not found!")

    # Check environments
    env_dir = tilling_dir / 'environments'
    if env_dir.exists():
        if (env_dir / 'environment_manifest.parquet').exists():
            print(f"\n  [✓] Loading environment manifest...")
            df_env = pd.read_parquet(env_dir / 'environment_manifest.parquet')
            print(f"    Total environments: {len(df_env)}")
            print(f"    Mean timesteps: {df_env['num_timesteps'].mean():.1f}")
            print(f"    Mean size: {df_env['height'].mean():.1f} x {df_env['width'].mean():.1f} cells")
        else:
            print(f"  [⚠️] WARNING: Environment manifest not found")
    else:
        print(f"  [⚠️] WARNING: Environments directory not found")

else:
    print("[✗] ERROR: Tilling data directory not found!")

# =============================================================================
# 3. CROSS-VALIDATION: FIRE DATA vs TILLED DATA
# =============================================================================
print("\n" + "=" * 80)
print("3. CROSS-VALIDATION: FIRE DATA vs TILLED DATA")
print("=" * 80)

if df_fire is not None and (tilling_dir / 'window_regions.parquet').exists():
    df_regions = pd.read_parquet(tilling_dir / 'window_regions.parquet')

    print("\n  Checking coordinate consistency...")

    # Compare coordinate ranges
    fire_x_min, fire_x_max = df_fire['x'].min(), df_fire['x'].max()
    fire_y_min, fire_y_max = df_fire['y'].min(), df_fire['y'].max()

    print(f"    Fire data spatial extent:")
    print(f"      X: [{fire_x_min:.2f}, {fire_x_max:.2f}]")
    print(f"      Y: [{fire_y_min:.2f}, {fire_y_max:.2f}]")

    # Sample a region and check if it aligns
    if len(df_regions) > 0:
        sample_region = np.load(tilling_dir / 'regions' / f'window_region_{df_regions.iloc[0]["window_id"]:05d}.npz', allow_pickle=True)
        wb = sample_region['world_bounds'].item()

        print(f"\n    Sample region world bounds:")
        print(f"      X: [{wb['x_min']:.2f}, {wb['x_max']:.2f}]")
        print(f"      Y: [{wb['y_min']:.2f}, {wb['y_max']:.2f}]")

        # Check if region is within fire data extent
        if (wb['x_min'] >= fire_x_min and wb['x_max'] <= fire_x_max and
            wb['y_min'] >= fire_y_min and wb['y_max'] <= fire_y_max):
            print(f"      [✓] Region is within fire data extent")
        else:
            print(f"      [⚠️] WARNING: Region may be outside fire data extent")

    # Check temporal consistency
    print("\n  Checking temporal consistency...")
    fire_time_min = df_fire['datetime'].min()
    fire_time_max = df_fire['datetime'].max()
    region_time_min = df_regions['time_start'].min()
    region_time_max = df_regions['time_end'].max()

    print(f"    Fire data time range: {fire_time_min} to {fire_time_max}")
    print(f"    Region data time range: {region_time_min} to {region_time_max}")

    if region_time_min >= fire_time_min and region_time_max <= fire_time_max:
        print(f"    [✓] Region times are within fire data time range")
    else:
        print(f"    [⚠️] WARNING: Region times may be outside fire data time range")

# =============================================================================
# 4. DETAILED WEATHER DATA ALIGNMENT CHECK
# =============================================================================
print("\n" + "=" * 80)
print("4. WEATHER DATA ALIGNMENT CHECK")
print("=" * 80)

if df_fire is not None and 'w' in df_fire.columns:
    print("\n  Analyzing weather data coverage...")

    # Check how many fire detections have weather data
    has_weather = df_fire[['w', 'rh', 'r']].notna().all(axis=1)
    coverage = 100 * has_weather.sum() / len(df_fire)
    print(f"    Fire detections with complete weather data: {has_weather.sum():,}/{len(df_fire):,} ({coverage:.2f}%)")

    if coverage < 100:
        print(f"    [⚠️] WARNING: {100-coverage:.2f}% of fire detections lack complete weather data")

        # Check temporal distribution of missing weather
        missing_weather = df_fire[~has_weather]
        if len(missing_weather) > 0:
            print(f"\n    Missing weather data time range:")
            print(f"      Earliest: {missing_weather['datetime'].min()}")
            print(f"      Latest: {missing_weather['datetime'].max()}")
    else:
        print(f"    [✓] All fire detections have complete weather data")

    # Check weather data statistics
    print(f"\n  Weather data statistics:")
    weather_stats = df_fire[['w', 'd_x', 'd_y', 'rh', 'r']].describe()
    print(weather_stats.to_string())

    # Check for suspicious values
    print(f"\n  Checking for suspicious weather values...")
    suspicious = []

    if 'w' in df_fire.columns:
        if (df_fire['w'] < 0).any() or (df_fire['w'] > 50).any():
            suspicious.append("Wind speed out of range [0, 50] m/s")

    if 'rh' in df_fire.columns:
        if (df_fire['rh'] < 0).any() or (df_fire['rh'] > 100).any():
            suspicious.append("Relative humidity out of range [0, 100]%")

    if 'r' in df_fire.columns:
        if (df_fire['r'] < 0).any():
            suspicious.append("Negative rainfall values")

    if suspicious:
        for issue in suspicious:
            print(f"    [⚠️] WARNING: {issue}")
    else:
        print(f"    [✓] No suspicious weather values detected")

else:
    print("  [⚠️] WARNING: Weather data not found in fire dataset")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

summary = {
    "filtered_fire_data_exists": df_fire is not None,
    "tilling_data_exists": tilling_dir.exists(),
    "regions_exist": (tilling_dir / 'regions').exists() if tilling_dir.exists() else False,
    "sequences_exist": (tilling_dir / 'sequences').exists() if tilling_dir.exists() else False,
    "environments_exist": (tilling_dir / 'environments').exists() if tilling_dir.exists() else False,
}

if df_fire is not None:
    summary["fire_detection_count"] = len(df_fire)
    summary["fire_has_coordinates"] = 'x' in df_fire.columns and 'y' in df_fire.columns
    summary["fire_has_datetime"] = 'datetime' in df_fire.columns
    summary["fire_has_weather"] = all(col in df_fire.columns for col in ['w', 'rh', 'r'])

print("\n  Status:")
for key, value in summary.items():
    status = "[✓]" if value else "[✗]"
    print(f"    {status} {key}: {value}")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
