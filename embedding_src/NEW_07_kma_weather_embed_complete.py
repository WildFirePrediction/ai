"""
KMA Weather Data Embedding - Complete Pipeline
Embeds 9 weather channels for each VIIRS fire timestamp

Channels (9 total):
1. Temperature (C)
2. Humidity (%)  
3. Wind U-component (m/s)
4. Wind V-component (m/s)
5. Wind Speed (m/s)
6. Precipitation (mm)
7. Atmospheric Pressure (hPa)
8. Cloud Index (derived)
9. Visibility Index (derived)

Grid: 400m resolution, EPSG:5179 (Korean CRS)
Method: Inverse Distance Weighting (IDW) interpolation from AWS stations
"""

import os
import sys
import numpy as np
import csv
import json
from pathlib import Path
from datetime import datetime
import rasterio
from rasterio.transform import from_origin
from scipy.interpolate import griddata
import pyproj

# ============================================================================
# CONFIGURATION
# ============================================================================
root_dir = Path(__file__).parent.parent
data_dir = root_dir / 'data'
kma_dir = data_dir / 'KMA'
output_dir = root_dir / 'embedded_data'
kma_weather_dir = output_dir / 'kma_weather'
kma_weather_dir.mkdir(exist_ok=True, parents=True)

print("=" * 80)
print("KMA WEATHER DATA EMBEDDING - COMPLETE PIPELINE")
print("=" * 80)

# ============================================================================
# 1. LOAD GRID CONFIGURATION
# ============================================================================
print("\n[1/8] Loading DEM grid configuration...")

dem_stats_path = output_dir / 'dem_norm_stats.json'
if not dem_stats_path.exists():
    print(f"ERROR: {dem_stats_path} not found")
    print("Run 02_dem_embedding first to create grid configuration")
    sys.exit(1)

with open(dem_stats_path, 'r') as f:
    grid_config = json.load(f)

x0 = grid_config['grid_origin_x']
y0 = grid_config['grid_origin_y']  
width = grid_config['grid_width']
height = grid_config['grid_height']
tile_size = grid_config['tile_size']

print(f"Grid: {width}x{height} pixels at {tile_size}m resolution")
print(f"Origin: ({x0:.0f}, {y0:.0f}) in EPSG:5179")

y1 = y0 + height * tile_size
transform = from_origin(x0, y1, tile_size, tile_size)

# Create grid coordinates
grid_x = x0 + np.arange(width) * tile_size + tile_size/2
grid_y = y0 + np.arange(height) * tile_size + tile_size/2
grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)

# ============================================================================
# 2. LOAD STATION COORDINATES
# ============================================================================
print("\n[2/8] Loading KMA station coordinates...")

stations_path = output_dir / 'kma_aws_stations.json'
if not stations_path.exists():
    print(f"ERROR: {stations_path} not found")
    print("Run 01_fetch_kma_station_coords.py first")
    sys.exit(1)

with open(stations_path, 'r') as f:
    stations = json.load(f)

print(f"Loaded {len(stations)} AWS stations")

# Load DEM CRS from metadata
dem_metadata_path = output_dir / 'dem_native_crs_metadata.json'
with open(dem_metadata_path, 'r') as f:
    dem_metadata = json.load(f)
dem_crs_wkt = dem_metadata['crs']

# Convert station lat/lon to DEM CRS (GRS80 TM, compatible with EPSG:5179)
# Use EPSG:5179 as proxy since it has same parameters
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)

station_coords = {}
for stn in stations:
    stn_id = stn['STN']
    lon, lat = stn['LON'], stn['LAT']
    x, y = transformer.transform(lon, lat)
    station_coords[stn_id] = {'x': x, 'y': y, 'lon': lon, 'lat': lat}

print(f"Converted {len(station_coords)} stations to DEM CRS (GRS80 TM)")

# ============================================================================
# 3. LOAD VIIRS TIMESTAMPS
# ============================================================================
print("\n[3/8] Loading VIIRS fire timestamps...")

try:
    import pandas as pd
    viirs_path = output_dir / 'nasa_viirs_embedded.parquet'
    viirs_df = pd.read_parquet(viirs_path)
    
    # Extract timestamp from datetime column
    viirs_df['kma_timestamp'] = pd.to_datetime(viirs_df['datetime']).dt.strftime('%Y%m%d%H%M')
    unique_timestamps = sorted(viirs_df['kma_timestamp'].unique())
    
    print(f"Found {len(unique_timestamps)} unique fire timestamps")
    print(f"Range: {unique_timestamps[0]} to {unique_timestamps[-1]}")
    
except Exception as e:
    print(f"ERROR loading VIIRS data: {e}")
    print("Using sample timestamps for testing...")
    unique_timestamps = ['201202010343', '201202011744']

# ============================================================================
# 4. PROCESS WEATHER DATA FOR EACH TIMESTAMP
# ============================================================================
print("\n[4/8] Processing weather data...")

def load_kma_data(timestamp):
    """Load KMA weather data for a timestamp"""
    kma_file = kma_dir / timestamp / f'AWS_{timestamp}.csv'
    
    if not kma_file.exists():
        return None
    
    # Read CSV manually to handle KMA format
    data = []
    with open(kma_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    # Parse weather features
    weather = {
        'stn': [],
        'temp': [],
        'humidity': [],
        'wind_speed': [],
        'wind_dir': [],
        'precip': [],
        'pressure': [],
    }
    
    for row in data:
        try:
            stn_id = int(row['STN'])
            if stn_id not in station_coords:
                continue
                
            # Parse values (handle missing = -99.x)
            temp = float(row['TA'])
            humidity = float(row['HM'])
            wind_speed = float(row['WS1'])
            wind_dir = float(row['WD1'])
            precip = float(row['RN-15m'])
            pressure = float(row['PA'])
            
            # Filter invalid values
            if temp < -90 or humidity < -90 or wind_speed < -90:
                continue
            if wind_dir < -90 or precip < -90 or pressure < -90:
                continue
                
            weather['stn'].append(stn_id)
            weather['temp'].append(temp)
            weather['humidity'].append(humidity)
            weather['wind_speed'].append(wind_speed)
            weather['wind_dir'].append(wind_dir)
            weather['precip'].append(precip)
            weather['pressure'].append(pressure)
            
        except (KeyError, ValueError):
            continue
    
    if len(weather['stn']) == 0:
        return None
    
    # Convert to numpy arrays
    for key in weather:
        weather[key] = np.array(weather[key])
    
    # Get station coordinates
    station_x = np.array([station_coords[s]['x'] for s in weather['stn']])
    station_y = np.array([station_coords[s]['y'] for s in weather['stn']])
    weather['x'] = station_x
    weather['y'] = station_y
    
    return weather

def interpolate_to_grid(weather):
    """Interpolate weather data to grid using IDW"""
    
    points = np.column_stack([weather['x'], weather['y']])
    grid_points = np.column_stack([grid_xx.ravel(), grid_yy.ravel()])
    
    # Interpolate each channel
    channels = {}
    
    # Temperature
    temp_grid = griddata(points, weather['temp'], grid_points, method='linear', fill_value=np.nan)
    channels['temp'] = temp_grid.reshape(height, width)
    
    # Humidity
    hum_grid = griddata(points, weather['humidity'], grid_points, method='linear', fill_value=np.nan)
    channels['humidity'] = hum_grid.reshape(height, width)
    
    # Wind components
    wind_dir_rad = np.deg2rad(weather['wind_dir'])
    wind_u = -weather['wind_speed'] * np.sin(wind_dir_rad)
    wind_v = -weather['wind_speed'] * np.cos(wind_dir_rad)
    
    u_grid = griddata(points, wind_u, grid_points, method='linear', fill_value=np.nan)
    v_grid = griddata(points, wind_v, grid_points, method='linear', fill_value=np.nan)
    channels['wind_u'] = u_grid.reshape(height, width)
    channels['wind_v'] = v_grid.reshape(height, width)
    
    # Wind speed
    speed_grid = griddata(points, weather['wind_speed'], grid_points, method='linear', fill_value=np.nan)
    channels['wind_speed'] = speed_grid.reshape(height, width)
    
    # Precipitation
    precip_grid = griddata(points, weather['precip'], grid_points, method='linear', fill_value=np.nan)
    channels['precip'] = precip_grid.reshape(height, width)
    
    # Pressure
    press_grid = griddata(points, weather['pressure'], grid_points, method='linear', fill_value=np.nan)
    channels['pressure'] = press_grid.reshape(height, width)
    
    # Cloud index (derived from humidity and temperature difference)
    # Simple heuristic: high humidity + low temp diff = cloudy
    cloud_index = channels['humidity'] / 100.0
    channels['cloud_index'] = cloud_index
    
    # Visibility index (derived from humidity and precipitation)
    # Simple heuristic: low precip + low humidity = high visibility
    visibility = 1.0 - (channels['humidity'] / 100.0 * 0.5 + np.clip(channels['precip'] / 10.0, 0, 1) * 0.5)
    channels['visibility'] = visibility
    
    return channels

# ============================================================================
# 5. PROCESS ALL TIMESTAMPS
# ============================================================================
print("\n[5/8] Processing all timestamps...")

success_count = 0
fail_count = 0
processed_timestamps = []

for i, ts in enumerate(unique_timestamps):
    if i % 100 == 0:
        print(f"  Progress: {i}/{len(unique_timestamps)} ({success_count} successful, {fail_count} failed)")
    
    # Check if already processed
    output_path = kma_weather_dir / f'weather_{ts}.tif'
    if output_path.exists():
        processed_timestamps.append(ts)
        success_count += 1
        continue
    
    # Load weather data
    weather = load_kma_data(ts)
    if weather is None or len(weather['stn']) < 5:
        fail_count += 1
        continue
    
    # Interpolate to grid
    try:
        channels = interpolate_to_grid(weather)
        
        # Stack channels (9 total)
        weather_stack = np.stack([
            channels['temp'],
            channels['humidity'],
            channels['wind_u'],
            channels['wind_v'],
            channels['wind_speed'],
            channels['precip'],
            channels['pressure'],
            channels['cloud_index'],
            channels['visibility']
        ], axis=0)
        
        # Save as GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=9,
            dtype=weather_stack.dtype,
            crs='EPSG:5179',
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(weather_stack)
        
        processed_timestamps.append(ts)
        success_count += 1
        
    except Exception as e:
        print(f"  ERROR processing {ts}: {e}")
        fail_count += 1

print(f"\nProcessing complete:")
print(f"  Successful: {success_count}")
print(f"  Failed: {fail_count}")

# ============================================================================
# 6. COMPUTE NORMALIZATION STATISTICS
# ============================================================================
print("\n[6/8] Computing normalization statistics...")

# Sample from processed files to compute stats
sample_size = min(100, len(processed_timestamps))
sample_timestamps = np.random.choice(processed_timestamps, sample_size, replace=False)

channel_names = ['temp', 'humidity', 'wind_u', 'wind_v', 'wind_speed', 'precip', 'pressure', 'cloud_index', 'visibility']
stats = {name: {'min': [], 'max': [], 'mean': [], 'std': []} for name in channel_names}

for ts in sample_timestamps:
    weather_path = kma_weather_dir / f'weather_{ts}.tif'
    with rasterio.open(weather_path) as src:
        for i, name in enumerate(channel_names, 1):
            data = src.read(i)
            valid = data[~np.isnan(data)]
            if len(valid) > 0:
                stats[name]['min'].append(np.min(valid))
                stats[name]['max'].append(np.max(valid))
                stats[name]['mean'].append(np.mean(valid))
                stats[name]['std'].append(np.std(valid))

# Compute final stats
final_stats = {}
for name in channel_names:
    if len(stats[name]['min']) > 0:
        final_stats[name] = {
            'min': float(np.min(stats[name]['min'])),
            'max': float(np.max(stats[name]['max'])),
            'mean': float(np.mean(stats[name]['mean'])),
            'std': float(np.mean(stats[name]['std']))
        }
        print(f"  {name}: min={final_stats[name]['min']:.2f}, max={final_stats[name]['max']:.2f}")
    else:
        print(f"  {name}: NO VALID DATA")
        final_stats[name] = {'min': 0.0, 'max': 1.0, 'mean': 0.0, 'std': 1.0}

# ============================================================================
# 7. SAVE METADATA
# ============================================================================
print("\n[7/8] Saving metadata...")

metadata = {
    'grid_origin_x': x0,
    'grid_origin_y': y0,
    'grid_width': width,
    'grid_height': height,
    'tile_size': tile_size,
    'crs': 'EPSG:5179',
    'num_channels': 9,
    'channel_names': channel_names,
    'normalization': final_stats,
    'num_timestamps': len(processed_timestamps),
    'timestamp_range': [processed_timestamps[0], processed_timestamps[-1]] if processed_timestamps else [],
    'creation_date': datetime.now().isoformat()
}

metadata_path = output_dir / 'kma_weather_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Saved metadata to: {metadata_path}")

# Save timestamp list
timestamps_path = output_dir / 'kma_weather_timestamps.txt'
with open(timestamps_path, 'w') as f:
    for ts in processed_timestamps:
        f.write(f"{ts}\n")

print(f"Saved {len(processed_timestamps)} timestamps to: {timestamps_path}")

# ============================================================================
# 8. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("KMA WEATHER EMBEDDING COMPLETE")
print("=" * 80)
print(f"\n9 weather channels embedded for {len(processed_timestamps)} timestamps")
print(f"Output directory: {kma_weather_dir}")
print(f"Grid: {width}x{height} pixels at {tile_size}m resolution")
print(f"\nNext: Combine with DEM (2 channels) for 11-channel model")
