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

    # KMA AWS data format has header row
    # Columns: TIME,STN,WD1,WS1,WDS,WSS,WD10,WS10,TA,RE,RN-15m,RN-60m,RN-12H,RN-DAY,HM,PA,PS,TD
    df_kma = pd.read_csv(kma_file)
    print(f"Loaded {len(df_kma)} station records")
    print(f"Columns: {df_kma.columns.tolist()}")

    # Rename columns for clarity
    df_kma = df_kma.rename(columns={
        'TIME': 'timestamp',
        'STN': 'station_id',
        'WD1': 'wind_dir',      # Wind direction (degrees)
        'WS1': 'wind_speed',     # Wind speed (m/s)
        'WD10': 'wind_dir_10',   # 10-min avg wind direction
        'WS10': 'wind_speed_10', # 10-min avg wind speed
        'TA': 'temp',            # Temperature (°C)
        'HM': 'humid',           # Humidity (%)
        'RN-60m': 'precip_60',   # 60-min precipitation (mm)
        'RN-DAY': 'precip_day',  # Daily precipitation (mm)
        'PA': 'pressure',        # Pressure (hPa)
        'TD': 'temp_dew'         # Dew point temperature (°C)
    })

    # Replace missing values (-99.9, -99) with NaN
    df_kma = df_kma.replace([-99.9, -99, -999], np.nan)

    print("\nData sample:")
    print(df_kma[['station_id', 'wind_speed', 'wind_dir', 'temp', 'humid', 'precip_60']].head())
else:
    print("ERROR: No KMA data files found!")
    exit(1)

# ============================================================================
# 3. LOAD STATION LOCATIONS
# ============================================================================
print("\n[3/6] Loading station locations...")

# KMA AWS station locations (official coordinates from KMA)
# Station ID: (latitude, longitude)
# Source: Korea Meteorological Administration AWS station metadata
KMA_AWS_STATIONS = {
    90: (37.5714, 126.9658),    # Seoul
    92: (35.1041, 129.0320),    # Busan
    93: (37.4563, 126.7052),    # Incheon
    95: (35.8714, 128.6014),    # Daegu
    98: (35.1595, 126.8526),    # Gwangju
    99: (35.5383, 129.3114),    # Ulsan
    100: (36.3667, 127.3833),   # Daejeon
    101: (36.6424, 127.4890),   # Sejong
    108: (37.9838, 127.0288),   # Uijeongbu
    112: (37.2636, 127.0286),   # Suwon
    114: (37.7480, 128.8760),   # Gangneung
    115: (37.8813, 127.7298),   # Chuncheon
    119: (35.8279, 127.1480),   # Jeonju
    121: (36.7995, 127.0057),   # Cheongju
    127: (35.1064, 126.8906),   # Mokpo
    129: (36.0190, 129.3435),   # Pohang
    130: (34.8161, 126.3916),   # Yeosu
    131: (35.8203, 127.1089),   # Gunsan
    133: (34.4736, 126.6228),   # Heuksando
    135: (37.1337, 128.2093),   # Wonju
    136: (33.5141, 126.5292),   # Jeju
    137: (33.2894, 126.5653),   # Seogwipo
    138: (33.3822, 126.2997),   # Gosan
    140: (37.2757, 126.4597),   # Suwon Air Base
    143: (37.9050, 127.0368),   # Dongducheon
    146: (37.1560, 128.4611),   # Jecheon
    152: (35.6053, 126.7158),   # Gunsan Air Base
    155: (35.1779, 129.0750),   # Busan Air Base
    156: (37.4531, 127.1286),   # Gwangju (Gyeonggi)
    159: (35.1174, 128.0802),   # Jinju
    162: (35.9886, 129.4261),   # Yangsan
    165: (37.2750, 127.4355),   # Icheon
    168: (37.7435, 128.7208),   # Sokcho
    170: (37.0878, 127.5806),   # Yeoju
    184: (37.9658, 126.6363),   # Paju
    185: (37.0895, 128.4571),   # Chungju
    189: (36.0365, 127.6366),   # Geumsan
    192: (36.2439, 127.8331),   # Boeun
    201: (37.2757, 127.0093),   # Yongin
    202: (36.8937, 127.1325),   # Cheongyang
    203: (35.4189, 127.3880),   # Namwon
    211: (37.9106, 127.7401),   # Inje
    212: (38.2072, 128.5944),   # Goseong
    216: (37.9736, 127.0622),   # Pocheon
    217: (38.1772, 128.4656),   # Yangyang
    221: (37.2393, 128.3586),   # Cheonan
    226: (36.6359, 127.4419),   # Boryeong
    232: (35.6840, 127.1506),   # Imsil
    235: (35.8205, 127.4892),   # Jangsu
    236: (35.5052, 127.0144),   # Sunchang
    238: (35.3819, 126.9525),   # Jangheung
    239: (35.2218, 126.8471),   # Haenam
    243: (34.8236, 126.3046),   # Goheung
    244: (34.6840, 127.7405),   # Yeosu Air Base
    245: (34.7462, 127.5064),   # Boseong
    247: (35.5793, 126.7111),   # Gangjin
    248: (34.7739, 127.9022),   # Jangheung Marine
    260: (35.1354, 128.6811),   # Changwon
    261: (35.5547, 129.3114),   # Yangsan
    262: (35.1013, 128.4750),   # Jinju Air Base
    271: (37.0063, 127.2654),   # Mungyeong
    272: (36.9213, 128.6231),   # Yeongju
    273: (37.6708, 129.1230),   # Uljin
    276: (36.5691, 128.7289),   # Bonghwa
    277: (35.4908, 129.1663),   # Miryang
    278: (35.3347, 129.0364),   # Sancheong
    279: (35.0681, 127.9094),   # Geochang
    281: (35.2288, 128.6917),   # Hapcheon
    283: (35.8383, 128.5569),   # Gumi
    284: (36.4200, 129.3658),   # Yeongdeok
    285: (36.0328, 129.3653),   # Gyeongju
    288: (37.4913, 129.1167),   # Donghae
    289: (37.8772, 128.7208),   # Taebaek
    294: (36.3731, 129.3667),   # Pohang Air Base
    295: (37.0630, 127.8797),   # Chungju Air Base
}

# Map station IDs to coordinates
station_ids = df_kma['station_id'].values
station_coords_list = []

missing_stations = set()
for stn_id in station_ids:
    if stn_id in KMA_AWS_STATIONS:
        lat, lon = KMA_AWS_STATIONS[stn_id]
        station_coords_list.append((lat, lon))
    else:
        # For missing stations, use None and handle later
        station_coords_list.append((None, None))
        missing_stations.add(stn_id)

if missing_stations:
    print(f"  WARNING: {len(missing_stations)} stations not found in coordinate database:")
    print(f"    Missing IDs: {sorted(list(missing_stations))[:20]}")  # Show first 20
    print(f"  These stations will be excluded from interpolation")

# Create lat/lon arrays, filtering out missing stations
valid_coords = [(lat, lon) for lat, lon in station_coords_list if lat is not None]
if not valid_coords:
    print("ERROR: No valid station coordinates found!")
    print("Please update the KMA_AWS_STATIONS dictionary with correct station IDs")
    exit(1)

# Filter dataframe to only include stations with known coordinates
valid_mask = [lat is not None for lat, lon in station_coords_list]
df_kma = df_kma[valid_mask].copy()
station_coords_list = [coord for coord in station_coords_list if coord[0] is not None]

df_kma['lat'] = [lat for lat, lon in station_coords_list]
df_kma['lon'] = [lon for lat, lon in station_coords_list]

print(f"  Loaded coordinates for {len(df_kma)} stations")
print(f"  Coordinate range: lat [{df_kma['lat'].min():.2f}, {df_kma['lat'].max():.2f}], "
      f"lon [{df_kma['lon'].min():.2f}, {df_kma['lon'].max():.2f}]")

# Transform to EPSG:5179 using modern Transformer API
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)

x_stations, y_stations = transformer.transform(
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
    mem_mb = grid_values.nbytes / 1024 / 1024
    print(f"    Range: [{grid_values.min():.4f}, {grid_values.max():.4f}] | Mem: {mem_mb:.1f}MB")

# Free intermediate grids
del X_grid, Y_grid, grid_coords, station_coords
import gc
gc.collect()

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

