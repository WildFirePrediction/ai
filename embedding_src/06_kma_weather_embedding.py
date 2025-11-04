"""
06 - KMA Weather Data Embedding
Processes Korea Meteorological Administration weather station data
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from pathlib import Path
from scipy.interpolate import griddata
import pyproj
import json

output_dir = Path('../embedded_data')
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("KMA WEATHER DATA EMBEDDING")
print("=" * 80)

# ============================================================================
# 1. LOAD GRID CONFIGURATION
# ============================================================================
print("\n[1/6] Loading grid configuration...")

stats_path = output_dir / 'dem_rsp_norm_stats.json'

if stats_path.exists():
    with open(stats_path, 'r') as f:
        grid_config = json.load(f)

    x0 = grid_config['grid_origin_x']
    y0 = grid_config['grid_origin_y']
    width = grid_config['grid_width']
    height = grid_config['grid_height']
    tile_size = grid_config['tile_size']

    print(f"Grid configuration loaded")
    print(f"  Origin: ({x0:.2f}, {y0:.2f})")
    print(f"  Dimensions: {width} x {height}")
    print(f"  Tile size: {tile_size}m")
else:
    print("ERROR: Grid configuration not found. Run 02_dem_rsp_embedding.py first.")
    exit(1)

target_crs = 'EPSG:5179'
y1 = y0 + height * tile_size
target_transform = from_origin(x0, y1, tile_size, tile_size)

# Create grid coordinates
x_grid = np.linspace(x0, x0 + width * tile_size, width)
y_grid = np.linspace(y0, y0 + height * tile_size, height)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# ============================================================================
# 2. LOAD KMA WEATHER DATA
# ============================================================================
print("\n[2/6] Loading KMA weather data...")

kma_dir = Path('../data/KMA')
csv_files = list(kma_dir.rglob('AWS_*.csv'))

print(f"Found {len(csv_files)} KMA AWS data files")

# For this example, we'll use one time period
# In practice, you'd want to match weather data to fire episodes temporally
if csv_files:
    # Load first available file
    kma_file = csv_files[0]
    print(f"Loading: {kma_file}")

    # KMA AWS data format (from data inspection):
    # Columns appear to be: timestamp, station_id, and various weather variables
    # The exact column mapping needs to be determined from documentation

    df_kma = pd.read_csv(kma_file, header=None)
    print(f"Loaded {len(df_kma)} station records")
    print(f"Columns: {len(df_kma.columns)}")

    # Based on common KMA AWS format, assign column names
    # This may need adjustment based on actual data format
    df_kma.columns = [
        'timestamp', 'station_id', 'wind_dir', 'wind_speed', 'wind_dir_10', 'wind_speed_10',
        'wind_dir_max', 'wind_speed_max', 'temp', 'temp_dew', 'humid', 'humid_min',
        'precip_15', 'precip_60', 'precip_day', 'pressure', 'pressure_sea', 'temp_ground'
    ]

    print("\nColumn names assigned (verify with KMA documentation)")
    print(df_kma.head())
else:
    print("ERROR: No KMA data files found!")
    exit(1)

# ============================================================================
# 3. LOAD STATION LOCATIONS
# ============================================================================
print("\n[3/6] Loading or creating station locations...")

# KMA station locations (these would normally come from a separate file)
# For now, we'll create synthetic locations within Korea
# In production, load actual station coordinates from KMA metadata

# Create sample station locations (you should replace with actual data)
station_lons = np.random.uniform(126, 130, len(df_kma))
station_lats = np.random.uniform(33, 38, len(df_kma))

df_kma['lon'] = station_lons
df_kma['lat'] = station_lats

print(f"Station locations assigned (replace with actual KMA station coordinates)")

# Transform to EPSG:5179
proj_src = pyproj.Proj("EPSG:4326")
proj_dst = pyproj.Proj("EPSG:5179")

x_stations, y_stations = pyproj.transform(
    proj_src,
    proj_dst,
    df_kma['lon'].values,
    df_kma['lat'].values
)

df_kma['x'] = x_stations
df_kma['y'] = y_stations

print(f"Coordinates transformed to EPSG:5179")

# ============================================================================
# 4. DECOMPOSE WIND DIRECTION
# ============================================================================
print("\n[4/6] Decomposing wind direction...")

# Wind direction decomposition: d_x = sin(direction), d_y = cos(direction)
wind_dir_rad = np.radians(df_kma['wind_dir'])
df_kma['wind_dir_x'] = np.sin(wind_dir_rad)
df_kma['wind_dir_y'] = np.cos(wind_dir_rad)

print("Wind direction decomposed to (d_x, d_y) components")

# ============================================================================
# 5. INTERPOLATE TO GRID
# ============================================================================
print("\n[5/6] Interpolating weather variables to grid...")

station_coords = np.column_stack([df_kma['x'].values, df_kma['y'].values])
grid_coords = np.column_stack([X_grid.flatten(), Y_grid.flatten()])

# Weather variables to interpolate
weather_vars = {
    'wind_speed': df_kma['wind_speed'].values,
    'wind_dir_x': df_kma['wind_dir_x'].values,
    'wind_dir_y': df_kma['wind_dir_y'].values,
    'humidity': df_kma['humid'].values,
    'precipitation': df_kma['precip_60'].values  # 60-minute precipitation
}

interpolated_data = {}

for var_name, values in weather_vars.items():
    print(f"  Interpolating {var_name}...")

    # Handle missing values
    valid_mask = ~np.isnan(values)
    if valid_mask.sum() == 0:
        print(f"    WARNING: No valid data for {var_name}, using zeros")
        interpolated_data[var_name] = np.zeros((height, width))
        continue

    grid_values = griddata(
        station_coords[valid_mask],
        values[valid_mask],
        (X_grid, Y_grid),
        method='linear',
        fill_value=np.nanmean(values[valid_mask])
    )

    interpolated_data[var_name] = grid_values
    print(f"    Range: [{grid_values.min():.4f}, {grid_values.max():.4f}]")

# ============================================================================
# 6. NORMALIZE AND SAVE
# ============================================================================
print("\n[6/6] Normalizing and saving...")

norm_stats = {}
normalized_data = {}

# Wind speed: z-score normalization
w = interpolated_data['wind_speed']
w_mean, w_std = w.mean(), w.std()
normalized_data['w_norm'] = (w - w_mean) / (w_std if w_std > 0 else 1.0)
norm_stats['wind_speed_mean'] = float(w_mean)
norm_stats['wind_speed_std'] = float(w_std)

# Wind direction components: already in [-1, 1]
normalized_data['d_x_norm'] = interpolated_data['wind_dir_x']
normalized_data['d_y_norm'] = interpolated_data['wind_dir_y']

# Humidity: z-score normalization
rh = interpolated_data['humidity']
rh_mean, rh_std = rh.mean(), rh.std()
normalized_data['rh_norm'] = (rh - rh_mean) / (rh_std if rh_std > 0 else 1.0)
norm_stats['humidity_mean'] = float(rh_mean)
norm_stats['humidity_std'] = float(rh_std)

# Precipitation: log transform then z-score
r = interpolated_data['precipitation']
r_log = np.log1p(r)
r_log_mean, r_log_std = r_log.mean(), r_log.std()
normalized_data['r_norm'] = (r_log - r_log_mean) / (r_log_std if r_log_std > 0 else 1.0)
norm_stats['precipitation_log_mean'] = float(r_log_mean)
norm_stats['precipitation_log_std'] = float(r_log_std)

print("Normalization completed")

# Save as multi-band GeoTIFF
output_path = output_dir / 'kma_weather_embedded.tif'

band_names = ['w_norm', 'd_x_norm', 'd_y_norm', 'rh_norm', 'r_norm']
num_bands = len(band_names)

output_meta = {
    'driver': 'GTiff',
    'dtype': 'float32',
    'width': width,
    'height': height,
    'count': num_bands,
    'crs': target_crs,
    'transform': target_transform,
    'compress': 'lzw',
    'tiled': True,
    'blockxsize': 256,
    'blockysize': 256
}

with rasterio.open(output_path, 'w', **output_meta) as dst:
    for i, band_name in enumerate(band_names, 1):
        dst.write(normalized_data[band_name].astype('float32'), i)
        dst.set_band_description(i, band_name)

print(f"Saved embedded data to: {output_path}")
print(f"Bands: {', '.join(band_names)}")
print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

# Save normalization statistics
norm_stats['num_stations'] = len(df_kma)
norm_stats['interpolation_method'] = 'linear'

stats_path = output_dir / 'kma_weather_norm_stats.json'
with open(stats_path, 'w') as f:
    json.dump(norm_stats, f, indent=2)

print(f"Saved normalization stats to: {stats_path}")

print("\n" + "=" * 80)
print("KMA WEATHER EMBEDDING COMPLETE")
print("=" * 80)
print(f"✓ Processed {len(df_kma)} weather stations")
print(f"✓ Interpolated to {width}x{height} grid")
print(f"✓ Ready for next stage")
print("\nNOTE: Replace synthetic station locations with actual KMA station coordinates!")

