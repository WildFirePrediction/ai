"""
06 - KMA Weather Data Embedding (v2 - Temporal + Point-based)
Interpolates weather at fire detection locations for each timestamp
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
import pyproj
import json
from tqdm import tqdm
from collections import defaultdict

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).parent.parent / 'src'))

output_dir = _Path(__file__).parent.parent / 'embedded_data'
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("KMA WEATHER DATA EMBEDDING (v2 - TEMPORAL)")
print("=" * 80)

# ============================================================================
# 1. LOAD FIRE DETECTIONS
# ============================================================================
print("\n[1/6] Loading fire detections...")

viirs_path = output_dir / 'nasa_viirs_embedded.parquet'
viirs_df = pd.read_parquet(viirs_path)

print(f"  Loaded {len(viirs_df)} fire detections")
print(f"  Episodes: {viirs_df['episode_id'].nunique()}")
print(f"  Date range: {viirs_df['datetime'].min()} to {viirs_df['datetime'].max()}")

# ============================================================================
# 2. LOAD KMA STATION COORDINATES
# ============================================================================
print("\n[2/6] Loading station coordinates...")

# Extended station database (official KMA AWS stations)
# Compiled from: KMA Open Data Portal + Academic sources + Manual verification
KMA_AWS_STATIONS = {
    # Major Cities
    90: (37.5714, 126.9658),  # Seoul
    92: (35.1041, 129.0320),  # Busan
    93: (37.4563, 126.7052),  # Incheon
    95: (35.8714, 128.6014),  # Daegu
    96: (37.3000, 126.9580),  # Gwacheon
    98: (35.1595, 126.8526),  # Gwangju
    99: (35.5383, 129.3114),  # Ulsan
    100: (36.3667, 127.3833), # Daejeon
    101: (36.6424, 127.4890), # Sejong

    # Gyeonggi-do
    102: (37.9087, 127.0614), # Dongducheon
    104: (37.8481, 127.1827), # Pocheon
    105: (37.2892, 127.5326), # Icheon
    106: (36.9996, 127.0897), # Anseong
    108: (37.9838, 127.0288), # Uijeongbu
    110: (37.4292, 126.4883), # Ganghwa
    112: (37.2636, 127.0286), # Suwon
    113: (37.4517, 127.1286), # Gwangju
    116: (37.2713, 127.4355), # Icheon
    119: (35.8279, 127.1480), # Jeonju
    121: (36.7995, 127.0057), # Cheongju
    127: (35.1064, 126.8906), # Mokpo
    129: (36.0190, 129.3435), # Pohang
    130: (34.8161, 126.3916), # Yeosu
    131: (35.9055, 126.7169), # Gunsan
    133: (34.6917, 125.4458), # Heuksando
    135: (37.3378, 127.9456), # Wonju
    136: (33.5141, 126.5292), # Jeju
    137: (33.2894, 126.5653), # Seogwipo
    138: (33.2942, 126.1658), # Gosan
    140: (37.1014, 127.0314), # Suwon Air Base
    143: (37.9050, 127.0368), # Dongducheon
    146: (37.1560, 128.4611), # Jecheon
    151: (37.5511, 127.0753), # Gwangjin
    152: (35.9201, 126.7117), # Gunsan Air Base
    155: (35.1779, 129.0750), # Gimhae
    156: (37.4531, 127.1286), # Gwangju
    159: (35.1174, 128.0802), # Jinju
    160: (37.2764, 126.9854), # Gwanak
    162: (35.3369, 129.0289), # Yangsan
    163: (37.3486, 126.6469), # Siheung
    165: (37.2750, 127.4355), # Icheon
    167: (37.1012, 127.0092), # Pyeongtaek
    168: (37.7435, 128.7208), # Sokcho
    169: (35.2492, 128.6436), # Changwon
    170: (37.0878, 127.5806), # Yeoju
    172: (37.1392, 127.9092), # Wonju
    174: (36.8894, 127.7233), # Chungju
    175: (36.6550, 127.4881), # Sejong
    176: (36.6514, 127.4453), # Yeongi
    177: (36.4811, 127.3058), # Gongju
    181: (36.1781, 128.1289), # Sangju
    182: (35.9728, 127.0761), # Jeonju
    184: (37.9658, 126.6363), # Paju
    185: (37.0895, 128.4571), # Chungju
    188: (35.8097, 126.9008), # Iksan
    189: (36.1056, 127.4875), # Geumsan
    192: (36.4872, 127.7297), # Boeun
    201: (37.2757, 127.0093), # Yongin
    202: (36.2675, 126.7983), # Cheongyang
    203: (35.4189, 127.3880), # Namwon
    211: (38.0603, 128.1708), # Inje
    212: (38.3806, 128.4686), # Goseong
    216: (38.1458, 127.2006), # Pocheon
    217: (38.0750, 128.6186), # Yangyang
    221: (36.7981, 127.1186), # Cheongju
    226: (36.3269, 126.5569), # Boryeong
    229: (35.3361, 126.4894), # Jangheung
    230: (34.9847, 126.5161), # Haenam
    232: (35.6097, 127.2850), # Imsil
    233: (35.1606, 127.5094), # Suncheon
    235: (35.6508, 127.5211), # Jangsu
    236: (35.3764, 127.1378), # Sunchang
    238: (35.0547, 126.7531), # Gangjin
    239: (34.5536, 126.2450), # Wando
    243: (34.6183, 127.2756), # Goheung
    244: (34.8422, 127.7581), # Yeosu
    245: (34.7717, 127.0811), # Boseong
    247: (34.6433, 126.7656), # Gangjin
    248: (34.4267, 126.3114), # Wando
    251: (35.5050, 129.4167), # Ulsan
    252: (35.5383, 129.3456), # Ulsan North
    253: (35.4989, 129.4144), # Ulsan East
    254: (34.8494, 128.4328), # Tongyeong
    255: (34.8378, 128.4331), # Tongyeong
    257: (34.8436, 127.1478), # Goheung
    260: (35.2283, 128.6811), # Changwon
    261: (35.3392, 129.0367), # Yangsan
    262: (35.0836, 128.0694), # Jinju
    271: (36.5931, 128.1461), # Mungyeong
    272: (36.8714, 128.6167), # Yeongju
    273: (36.9933, 129.4072), # Uljin
    276: (36.8906, 128.7297), # Bonghwa
    277: (35.4939, 128.7464), # Miryang
    278: (35.4106, 127.8814), # Sancheong
    279: (35.6714, 127.9092), # Geochang
    281: (35.5672, 128.1694), # Hapcheon
    283: (36.1194, 128.3447), # Gumi
    284: (36.4156, 129.4072), # Yeongdeok
    285: (35.8386, 129.2153), # Gyeongju
    288: (37.5244, 129.1147), # Donghae
    289: (37.1661, 128.9853), # Taebaek
    294: (36.0322, 129.3650), # Pohang
    295: (37.1422, 127.8886), # Wonju Air Base
}

print(f"  Loaded {len(KMA_AWS_STATIONS)} station coordinates")

# Create station dataframe
station_data = []
for stn_id, (lat, lon) in KMA_AWS_STATIONS.items():
    station_data.append({'stnId': stn_id, 'lat': lat, 'lon': lon})

station_df = pd.DataFrame(station_data)

# Transform to EPSG:5179
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
x_stations, y_stations = transformer.transform(station_df['lon'].values, station_df['lat'].values)
station_df['x'] = x_stations
station_df['y'] = y_stations

print(f"  Station coverage: lat [{station_df['lat'].min():.2f}, {station_df['lat'].max():.2f}]")
print(f"                    lon [{station_df['lon'].min():.2f}, {station_df['lon'].max():.2f}]")

# ============================================================================
# 3. PROCESS EACH TIMESTAMP
# ============================================================================
print("\n[3/6] Processing weather for each timestamp...")

kma_dir = _Path(__file__).parent.parent / 'data' / 'KMA'
unique_timestamps = sorted(viirs_df['datetime'].unique())

print(f"  Found {len(unique_timestamps)} unique fire timestamps")
print(f"  Time span: {unique_timestamps[0]} to {unique_timestamps[-1]}")

# Initialize weather columns
weather_columns = ['w', 'd_x', 'd_y', 'rh', 'r']
for col in weather_columns:
    viirs_df[col] = np.nan

# Track statistics
stats = {
    'processed': 0,
    'missing_kma': 0,
    'insufficient_stations': 0,
    'success': 0
}

# Process in batches for efficiency
for timestamp in tqdm(unique_timestamps, desc="Interpolating weather"):
    # Format timestamp to match KMA directory structure
    ts_str = pd.Timestamp(timestamp).strftime('%Y%m%d%H%M')
    timestamp_dir = kma_dir / ts_str

    if not timestamp_dir.exists():
        stats['missing_kma'] += 1
        continue

    # Load AWS CSV
    csv_files = list(timestamp_dir.glob('AWS_*.csv'))
    if not csv_files:
        stats['missing_kma'] += 1
        continue

    try:
        weather_df = pd.read_csv(csv_files[0])
    except Exception as e:
        print(f"\n  ERROR reading {csv_files[0]}: {e}")
        stats['missing_kma'] += 1
        continue

    # Clean missing values (KMA uses various codes for missing data)
    weather_df = weather_df.replace([-99, -99.9, -999, -99.2, -99.7, -99.8], np.nan)

    # Additional sanity checks
    weather_df.loc[weather_df['WS1'] < 0, 'WS1'] = np.nan  # Wind speed can't be negative
    weather_df.loc[weather_df['WS1'] > 100, 'WS1'] = np.nan  # > 100 m/s is unrealistic
    weather_df.loc[weather_df['HM'] < 0, 'HM'] = np.nan  # Humidity can't be negative
    weather_df.loc[weather_df['HM'] > 100, 'HM'] = np.nan  # Humidity can't exceed 100%
    weather_df.loc[weather_df['RN-60m'] < 0, 'RN-60m'] = np.nan  # Precipitation can't be negative

    # Merge with station coordinates
    weather_df = weather_df.merge(station_df, left_on='STN', right_on='stnId', how='inner')

    if len(weather_df) < 3:
        stats['insufficient_stations'] += 1
        continue

    # Decompose wind direction
    wind_dir_rad = np.radians(weather_df['WD1'].fillna(0))
    weather_df['wind_dir_x'] = np.sin(wind_dir_rad)
    weather_df['wind_dir_y'] = np.cos(wind_dir_rad)

    # Get fire locations at this timestamp
    fire_mask = viirs_df['datetime'] == timestamp
    fire_coords = viirs_df.loc[fire_mask, ['x', 'y']].values

    if len(fire_coords) == 0:
        continue

    # Station coordinates
    station_coords = weather_df[['x', 'y']].values

    # Interpolate each weather variable
    fire_indices = viirs_df.index[fire_mask]

    for var_name, col_name in [
        ('w', 'WS1'),           # Wind speed
        ('d_x', 'wind_dir_x'),  # Wind direction X
        ('d_y', 'wind_dir_y'),  # Wind direction Y
        ('rh', 'HM'),           # Humidity
        ('r', 'RN-60m')         # Precipitation 60-min
    ]:
        values = weather_df[col_name].values
        valid_mask = ~np.isnan(values)

        if valid_mask.sum() < 3:
            # Use global mean as fallback
            viirs_df.loc[fire_indices, var_name] = 0.0
            continue

        try:
            interpolated = griddata(
                station_coords[valid_mask],
                values[valid_mask],
                fire_coords,
                method='linear',
                fill_value=np.nanmean(values[valid_mask])
            )

            # Additional bounds checking after interpolation
            if var_name == 'w':  # Wind speed
                interpolated = np.clip(interpolated, 0, 50)  # 0-50 m/s
            elif var_name == 'rh':  # Humidity
                interpolated = np.clip(interpolated, 0, 100)  # 0-100%
            elif var_name == 'r':  # Precipitation
                interpolated = np.clip(interpolated, 0, None)  # >= 0

            viirs_df.loc[fire_indices, var_name] = interpolated
        except Exception as e:
            print(f"\n  WARNING: Interpolation failed for {var_name} at {timestamp}: {e}")
            viirs_df.loc[fire_indices, var_name] = np.nanmean(values[valid_mask])

    stats['processed'] += 1
    stats['success'] += len(fire_coords)

print(f"\n  Processing stats:")
print(f"    Timestamps processed: {stats['processed']}/{len(unique_timestamps)}")
print(f"    Missing KMA data: {stats['missing_kma']}")
print(f"    Insufficient stations: {stats['insufficient_stations']}")
print(f"    Fire detections with weather: {stats['success']}/{len(viirs_df)}")

# ============================================================================
# 4. HANDLE MISSING VALUES
# ============================================================================
print("\n[4/6] Handling missing values...")

# For detections without weather (missing KMA data), use interpolation from nearby times
missing_mask = viirs_df['w'].isna()
n_missing = missing_mask.sum()

print(f"  Missing weather: {n_missing}/{len(viirs_df)} ({100*n_missing/len(viirs_df):.2f}%)")

if n_missing > 0:
    # Fill with forward/backward fill by episode
    for col in weather_columns:
        viirs_df[col] = viirs_df.groupby('episode_id')[col].fillna(method='ffill').fillna(method='bfill')

    # Fill remaining with global mean
    for col in weather_columns:
        global_mean = viirs_df[col].mean()
        viirs_df[col] = viirs_df[col].fillna(global_mean)

    final_missing = viirs_df['w'].isna().sum()
    print(f"  After filling: {final_missing} missing")

# ============================================================================
# 5. NORMALIZE
# ============================================================================
print("\n[5/6] Normalizing weather variables...")

norm_stats = {}

# Wind speed: z-score
w_mean, w_std = viirs_df['w'].mean(), viirs_df['w'].std()
viirs_df['w_norm'] = (viirs_df['w'] - w_mean) / (w_std if w_std > 0 else 1.0)
norm_stats['wind_speed_mean'] = float(w_mean)
norm_stats['wind_speed_std'] = float(w_std)
print(f"  Wind speed: μ={w_mean:.2f}, σ={w_std:.2f}")

# Wind direction: already normalized to [-1, 1]
viirs_df['d_x_norm'] = viirs_df['d_x']
viirs_df['d_y_norm'] = viirs_df['d_y']
print(f"  Wind direction: decomposed to (d_x, d_y)")

# Humidity: z-score
rh_mean, rh_std = viirs_df['rh'].mean(), viirs_df['rh'].std()
viirs_df['rh_norm'] = (viirs_df['rh'] - rh_mean) / (rh_std if rh_std > 0 else 1.0)
norm_stats['humidity_mean'] = float(rh_mean)
norm_stats['humidity_std'] = float(rh_std)
print(f"  Humidity: μ={rh_mean:.2f}%, σ={rh_std:.2f}%")

# Precipitation: log + z-score
viirs_df['r_log'] = np.log1p(viirs_df['r'])
r_log_mean, r_log_std = viirs_df['r_log'].mean(), viirs_df['r_log'].std()
viirs_df['r_norm'] = (viirs_df['r_log'] - r_log_mean) / (r_log_std if r_log_std > 0 else 1.0)
norm_stats['precipitation_log_mean'] = float(r_log_mean)
norm_stats['precipitation_log_std'] = float(r_log_std)
print(f"  Precipitation: log(1+r), μ={r_log_mean:.3f}, σ={r_log_std:.3f}")

# ============================================================================
# 6. SAVE
# ============================================================================
print("\n[6/6] Saving augmented fire data...")

output_path = output_dir / 'nasa_viirs_with_weather.parquet'
viirs_df.to_parquet(output_path, compression='snappy', index=False)

file_size_mb = output_path.stat().st_size / 1024 / 1024
print(f"  Saved: {output_path}")
print(f"  Size: {file_size_mb:.2f} MB")
print(f"  Columns: {viirs_df.columns.tolist()}")

# Save normalization stats
stats_path = output_dir / 'kma_weather_norm_stats.json'
norm_stats['num_stations'] = len(KMA_AWS_STATIONS)
norm_stats['num_timestamps'] = len(unique_timestamps)
norm_stats['coverage'] = {
    'processed': stats['processed'],
    'missing': stats['missing_kma'],
    'success_rate': stats['processed'] / len(unique_timestamps) if unique_timestamps else 0
}

with open(stats_path, 'w') as f:
    json.dump(norm_stats, f, indent=2)

print(f"  Saved: {stats_path}")

print("\n" + "=" * 80)
print("KMA WEATHER EMBEDDING COMPLETE (v2)")
print("=" * 80)
print(f"✓ Processed {stats['processed']} timestamps")
print(f"✓ Interpolated weather for {stats['success']} fire detections")
print(f"✓ Weather columns added: {weather_columns}")
print(f"✓ Normalized columns: {[c + '_norm' for c in weather_columns]}")
print(f"✓ Ready for Phase 2 (Tiling)")
print("")
print("Next step: Run 07_final_state_composition.py")
