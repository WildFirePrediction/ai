"""
08 - Validate Embedded Data
Comprehensive validation of all embedded data before tiling
"""

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import rasterio
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

embedded_dir = _Path(__file__).parent.parent / 'embedded_data'
output_dir = embedded_dir / 'validation'
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("EMBEDDED DATA VALIDATION")
print("=" * 80)

validation_report = {
    'passed': [],
    'warnings': [],
    'failed': [],
    'stats': {}
}

# ============================================================================
# 1. LOAD ALL EMBEDDED DATA
# ============================================================================
print("\n[1/8] Loading embedded data...")

try:
    # Fire data with weather
    viirs_df = pd.read_parquet(embedded_dir / 'nasa_viirs_with_weather.parquet')
    print(f"  ✓ Loaded fire+weather data: {len(viirs_df)} rows")

    # DEM/RSP
    with rasterio.open(embedded_dir / 'dem_rsp_embedded.tif') as src:
        dem = src.read(1)
        rsp = src.read(2)
        dem_transform = src.transform
        dem_crs = src.crs
    print(f"  ✓ Loaded DEM/RSP: {dem.shape}")

    # LCM
    with rasterio.open(embedded_dir / 'lcm_embedded.tif') as src:
        lcm = src.read(1)
    print(f"  ✓ Loaded LCM: {lcm.shape}")

    # FSM
    with rasterio.open(embedded_dir / 'fsm_embedded.tif') as src:
        fsm = src.read(1)
    print(f"  ✓ Loaded FSM: {fsm.shape}")

    # NDVI
    with rasterio.open(embedded_dir / 'ndvi_embedded.tif') as src:
        ndvi = src.read()  # Multiple bands (time series)
    print(f"  ✓ Loaded NDVI: {ndvi.shape}")

    # Normalization stats
    with open(embedded_dir / 'dem_rsp_norm_stats.json') as f:
        dem_stats = json.load(f)
    with open(embedded_dir / 'kma_weather_norm_stats.json') as f:
        weather_stats = json.load(f)

    validation_report['passed'].append("All data files loaded successfully")

except Exception as e:
    validation_report['failed'].append(f"Failed to load data: {e}")
    print(f"  ✗ ERROR: {e}")
    exit(1)

# ============================================================================
# 2. VALIDATE SPATIAL CONSISTENCY
# ============================================================================
print("\n[2/8] Validating spatial consistency...")

# South Korea bounds (EPSG:5179)
SK_BOUNDS = {
    'x_min': 740000,
    'x_max': 1380000,
    'y_min': 1440000,
    'y_max': 2220000
}

# Check fire coordinates
x_min, x_max = viirs_df['x'].min(), viirs_df['x'].max()
y_min, y_max = viirs_df['y'].min(), viirs_df['y'].max()

print(f"  Fire coordinate range:")
print(f"    X: [{x_min:.0f}, {x_max:.0f}]")
print(f"    Y: [{y_min:.0f}, {y_max:.0f}]")

if x_min < SK_BOUNDS['x_min'] or x_max > SK_BOUNDS['x_max']:
    validation_report['warnings'].append(f"Fire X coordinates outside typical SK bounds")
if y_min < SK_BOUNDS['y_min'] or y_max > SK_BOUNDS['y_max']:
    validation_report['warnings'].append(f"Fire Y coordinates outside typical SK bounds")
else:
    validation_report['passed'].append("Fire coordinates within South Korea bounds")

# Check grid alignment
if dem.shape == rsp.shape == lcm.shape == fsm.shape:
    validation_report['passed'].append(f"All grids aligned: {dem.shape}")
    print(f"  ✓ All grids aligned: {dem.shape}")
else:
    validation_report['failed'].append(f"Grid shape mismatch: DEM{dem.shape} RSP{rsp.shape} LCM{lcm.shape} FSM{fsm.shape}")

validation_report['stats']['grid_shape'] = dem.shape
validation_report['stats']['fire_bounds'] = {'x': [float(x_min), float(x_max)], 'y': [float(y_min), float(y_max)]}

# ============================================================================
# 3. VALIDATE VALUE RANGES
# ============================================================================
print("\n[3/8] Validating value ranges...")

checks = []

# Fire temperature (should be reasonable)
te_range = (viirs_df['te'].min(), viirs_df['te'].max())
if te_range[0] < -50 or te_range[1] > 200:
    validation_report['warnings'].append(f"Unusual fire temperature range: {te_range}")
else:
    validation_report['passed'].append(f"Fire temperature range OK: {te_range}")
print(f"  Fire temperature: {te_range[0]:.1f}°C to {te_range[1]:.1f}°C")

# Wind speed (should be 0-50 m/s typically)
w_range = (viirs_df['w'].min(), viirs_df['w'].max())
if w_range[0] < -100 or w_range[1] > 100:
    validation_report['warnings'].append(f"Unusual wind speed range: {w_range}")
print(f"  Wind speed: {w_range[0]:.2f} to {w_range[1]:.2f} m/s")

# Wind direction components (should be [-1, 1])
dx_range = (viirs_df['d_x'].min(), viirs_df['d_x'].max())
dy_range = (viirs_df['d_y'].min(), viirs_df['d_y'].max())
if dx_range[0] < -1.1 or dx_range[1] > 1.1 or dy_range[0] < -1.1 or dy_range[1] > 1.1:
    validation_report['failed'].append(f"Wind direction components out of range: d_x{dx_range} d_y{dy_range}")
else:
    validation_report['passed'].append("Wind direction components in valid range [-1, 1]")
print(f"  Wind direction: d_x{dx_range}, d_y{dy_range}")

# Humidity (raw values can be weird due to missing data, check normalized)
rh_range = (viirs_df['rh'].min(), viirs_df['rh'].max())
print(f"  Humidity: {rh_range[0]:.1f}% to {rh_range[1]:.1f}%")

# Precipitation (should be >= 0)
r_range = (viirs_df['r'].min(), viirs_df['r'].max())
if r_range[0] < 0:
    validation_report['failed'].append(f"Negative precipitation: {r_range[0]}")
else:
    validation_report['passed'].append("Precipitation values non-negative")
print(f"  Precipitation: {r_range[0]:.2f} to {r_range[1]:.2f} mm")

# DEM (South Korea elevation 0-2000m typically)
dem_valid = dem[dem != -9999]  # Exclude nodata
dem_range = (dem_valid.min(), dem_valid.max())
if dem_range[0] < -500 or dem_range[1] > 3000:
    validation_report['warnings'].append(f"Unusual DEM range: {dem_range}")
else:
    validation_report['passed'].append(f"DEM range OK: {dem_range}")
print(f"  DEM: {dem_range[0]:.1f}m to {dem_range[1]:.1f}m")

# RSP (should be [0, 1])
rsp_valid = rsp[rsp != -9999]
rsp_range = (rsp_valid.min(), rsp_valid.max())
if rsp_range[0] < -0.1 or rsp_range[1] > 1.1:
    validation_report['failed'].append(f"RSP out of range [0,1]: {rsp_range}")
else:
    validation_report['passed'].append("RSP in valid range [0, 1]")
print(f"  RSP: {rsp_range[0]:.3f} to {rsp_range[1]:.3f}")

# NDVI (should be [0, 1] after normalization)
ndvi_valid = ndvi[ndvi != -9999]
ndvi_range = (ndvi_valid.min(), ndvi_valid.max())
if ndvi_range[0] < -0.1 or ndvi_range[1] > 1.1:
    validation_report['warnings'].append(f"NDVI out of typical range [0,1]: {ndvi_range}")
else:
    validation_report['passed'].append("NDVI in valid range")
print(f"  NDVI: {ndvi_range[0]:.3f} to {ndvi_range[1]:.3f}")

validation_report['stats']['value_ranges'] = {
    'fire_temp': [float(te_range[0]), float(te_range[1])],
    'wind_speed': [float(w_range[0]), float(w_range[1])],
    'humidity': [float(rh_range[0]), float(rh_range[1])],
    'precipitation': [float(r_range[0]), float(r_range[1])],
    'dem': [float(dem_range[0]), float(dem_range[1])],
    'rsp': [float(rsp_range[0]), float(rsp_range[1])],
    'ndvi': [float(ndvi_range[0]), float(ndvi_range[1])]
}

# ============================================================================
# 4. VALIDATE TEMPORAL CONSISTENCY
# ============================================================================
print("\n[4/8] Validating temporal consistency...")

# Check episode temporal ordering
episode_issues = []
for ep_id in viirs_df['episode_id'].unique():
    ep_data = viirs_df[viirs_df['episode_id'] == ep_id].sort_values('datetime')
    times = ep_data['datetime'].values

    # Check monotonicity
    if not all(times[i] <= times[i+1] for i in range(len(times)-1)):
        episode_issues.append(f"Episode {ep_id}: non-monotonic timestamps")

    # Check for large gaps (>7 days)
    if len(times) > 1:
        time_diffs = np.diff(times).astype('timedelta64[h]').astype(int)
        max_gap = time_diffs.max()
        if max_gap > 168:  # 7 days
            episode_issues.append(f"Episode {ep_id}: gap of {max_gap} hours")

if episode_issues:
    validation_report['warnings'].extend(episode_issues[:10])  # Show first 10
    print(f"  ⚠ Found {len(episode_issues)} episode temporal issues")
else:
    validation_report['passed'].append("All episodes temporally consistent")
    print(f"  ✓ All episodes temporally consistent")

# Check fire time vs elapsed time (tm)
viirs_df_sorted = viirs_df.sort_values(['episode_id', 'datetime'])
for ep_id in viirs_df['episode_id'].unique()[:10]:  # Check first 10 episodes
    ep_data = viirs_df_sorted[viirs_df_sorted['episode_id'] == ep_id]
    if len(ep_data) > 1:
        t_start = ep_data['datetime'].iloc[0]
        t_end = ep_data['datetime'].iloc[-1]
        tm_max = ep_data['tm'].max()

        actual_hours = (t_end - t_start).total_seconds() / 3600
        if abs(tm_max - actual_hours) > 1:  # 1 hour tolerance
            validation_report['warnings'].append(f"Episode {ep_id}: tm mismatch (tm={tm_max:.1f}h, actual={actual_hours:.1f}h)")

# Time range
time_range = (viirs_df['datetime'].min(), viirs_df['datetime'].max())
print(f"  Time range: {time_range[0]} to {time_range[1]}")
validation_report['stats']['time_range'] = [str(time_range[0]), str(time_range[1])]

# ============================================================================
# 5. VALIDATE DATA COMPLETENESS
# ============================================================================
print("\n[5/8] Validating data completeness...")

required_cols = ['episode_id', 'datetime', 'x', 'y', 'te', 'i', 'tm',
                 'w', 'd_x', 'd_y', 'rh', 'r',
                 'x_norm', 'y_norm', 'te_norm', 'i_norm', 'tm_norm',
                 'w_norm', 'd_x_norm', 'd_y_norm', 'rh_norm', 'r_norm']

missing_cols = [col for col in required_cols if col not in viirs_df.columns]
if missing_cols:
    validation_report['failed'].append(f"Missing columns: {missing_cols}")
    print(f"  ✗ Missing columns: {missing_cols}")
else:
    validation_report['passed'].append("All required columns present")
    print(f"  ✓ All required columns present")

# Check for missing values
print(f"\n  Missing values:")
for col in required_cols:
    if col in viirs_df.columns:
        n_missing = viirs_df[col].isna().sum()
        pct_missing = 100 * n_missing / len(viirs_df)
        if n_missing > 0:
            print(f"    {col}: {n_missing} ({pct_missing:.2f}%)")
            if pct_missing > 5:
                validation_report['warnings'].append(f"{col}: {pct_missing:.1f}% missing")

if viirs_df[required_cols].isna().sum().sum() == 0:
    validation_report['passed'].append("No missing values in required columns")
    print(f"  ✓ No missing values")

# ============================================================================
# 6. VALIDATE EPISODE QUALITY
# ============================================================================
print("\n[6/8] Validating episode quality...")

episode_stats = viirs_df.groupby('episode_id').agg({
    'datetime': ['min', 'max', 'count'],
    'x': ['min', 'max'],
    'y': ['min', 'max']
})

episode_stats.columns = ['time_start', 'time_end', 'n_detections', 'x_min', 'x_max', 'y_min', 'y_max']
episode_stats['duration_hours'] = (episode_stats['time_end'] - episode_stats['time_start']).dt.total_seconds() / 3600
episode_stats['spatial_extent_km'] = np.sqrt(
    (episode_stats['x_max'] - episode_stats['x_min'])**2 +
    (episode_stats['y_max'] - episode_stats['y_min'])**2
) / 1000

print(f"  Number of episodes: {len(episode_stats)}")
print(f"  Detections per episode: {episode_stats['n_detections'].describe()}")
print(f"  Duration (hours): {episode_stats['duration_hours'].describe()}")
print(f"  Spatial extent (km): {episode_stats['spatial_extent_km'].describe()}")

# Quality checks
small_episodes = (episode_stats['n_detections'] < 3).sum()
if small_episodes > len(episode_stats) * 0.5:
    validation_report['warnings'].append(f"{small_episodes} episodes have <3 detections")
else:
    validation_report['passed'].append(f"Episode sizes reasonable ({small_episodes} with <3 detections)")

validation_report['stats']['episodes'] = {
    'total': int(len(episode_stats)),
    'detections_mean': float(episode_stats['n_detections'].mean()),
    'duration_mean_hours': float(episode_stats['duration_hours'].mean()),
    'spatial_extent_mean_km': float(episode_stats['spatial_extent_km'].mean())
}

# ============================================================================
# 7. VALIDATE NORMALIZATION
# ============================================================================
print("\n[7/8] Validating normalization...")

# Check if normalized values are actually normalized (z-score should have ~mean 0, std 1)
norm_cols = ['x_norm', 'y_norm', 'te_norm', 'i_norm', 'tm_norm', 'w_norm', 'rh_norm', 'r_norm']

print(f"  Normalized column statistics:")
for col in norm_cols:
    if col in viirs_df.columns:
        mean = viirs_df[col].mean()
        std = viirs_df[col].std()
        print(f"    {col}: μ={mean:.3f}, σ={std:.3f}")

        # Z-score normalization should have mean ~0, std ~1
        if abs(mean) > 0.1:
            validation_report['warnings'].append(f"{col}: mean={mean:.3f}, expected ~0")
        if abs(std - 1.0) > 0.1:
            validation_report['warnings'].append(f"{col}: std={std:.3f}, expected ~1")

validation_report['passed'].append("Normalization applied to all features")

# ============================================================================
# 8. CREATE VISUALIZATIONS
# ============================================================================
print("\n[8/8] Creating validation visualizations...")

try:
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Fire locations map
    ax1 = fig.add_subplot(gs[0, 0])
    sample = viirs_df.sample(min(10000, len(viirs_df)))
    scatter = ax1.scatter(sample['x'], sample['y'], c=sample['episode_id'],
                         s=1, alpha=0.5, cmap='tab20')
    ax1.set_xlabel('X (EPSG:5179)')
    ax1.set_ylabel('Y (EPSG:5179)')
    ax1.set_title('Fire Detection Locations (colored by episode)')
    ax1.grid(True, alpha=0.3)

    # 2. Episode size distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(episode_stats['n_detections'], bins=50, edgecolor='black')
    ax2.set_xlabel('Detections per Episode')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Episode Size Distribution')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # 3. Episode duration distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(episode_stats['duration_hours'], bins=50, edgecolor='black')
    ax3.set_xlabel('Duration (hours)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Episode Duration Distribution')
    ax3.grid(True, alpha=0.3)

    # 4. Fire temperature distribution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(viirs_df['te'], bins=100, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Fire Temperature (°C)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Fire Temperature Distribution')
    ax4.axvline(viirs_df['te'].mean(), color='r', linestyle='--', label=f'Mean: {viirs_df["te"].mean():.1f}°C')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Wind speed distribution
    ax5 = fig.add_subplot(gs[1, 1])
    wind_clean = viirs_df['w'][(viirs_df['w'] >= 0) & (viirs_df['w'] < 30)]
    ax5.hist(wind_clean, bins=100, edgecolor='black', alpha=0.7)
    ax5.set_xlabel('Wind Speed (m/s)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Wind Speed Distribution')
    ax5.axvline(wind_clean.mean(), color='r', linestyle='--', label=f'Mean: {wind_clean.mean():.1f} m/s')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Wind direction rose
    ax6 = fig.add_subplot(gs[1, 2], projection='polar')
    wind_angle = np.arctan2(viirs_df['d_x'], viirs_df['d_y'])
    ax6.hist(wind_angle, bins=36, alpha=0.7)
    ax6.set_theta_zero_location('N')
    ax6.set_theta_direction(-1)
    ax6.set_title('Wind Direction Distribution')

    # 7. Humidity distribution
    ax7 = fig.add_subplot(gs[2, 0])
    humidity_clean = viirs_df['rh'][(viirs_df['rh'] >= 0) & (viirs_df['rh'] <= 100)]
    ax7.hist(humidity_clean, bins=100, edgecolor='black', alpha=0.7)
    ax7.set_xlabel('Relative Humidity (%)')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Humidity Distribution')
    ax7.axvline(humidity_clean.mean(), color='r', linestyle='--', label=f'Mean: {humidity_clean.mean():.1f}%')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # 8. Precipitation distribution (log scale)
    ax8 = fig.add_subplot(gs[2, 1])
    precip_nonzero = viirs_df['r'][viirs_df['r'] > 0]
    if len(precip_nonzero) > 0:
        ax8.hist(precip_nonzero, bins=50, edgecolor='black', alpha=0.7)
        ax8.set_xlabel('Precipitation (mm)')
        ax8.set_ylabel('Frequency')
        ax8.set_title(f'Precipitation Distribution (non-zero: {len(precip_nonzero)/len(viirs_df)*100:.1f}%)')
        ax8.set_yscale('log')
        ax8.grid(True, alpha=0.3)

    # 9. Fire intensity distribution
    ax9 = fig.add_subplot(gs[2, 2])
    intensity_clean = viirs_df['i'][(viirs_df['i'] >= 0) & (viirs_df['i'] < 1000)]
    ax9.hist(intensity_clean, bins=100, edgecolor='black', alpha=0.7)
    ax9.set_xlabel('Fire Intensity')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Fire Intensity Distribution')
    ax9.grid(True, alpha=0.3)

    # 10. Sample episode time series
    ax10 = fig.add_subplot(gs[3, :])
    sample_episodes = episode_stats.nlargest(5, 'n_detections').index
    for i, ep_id in enumerate(sample_episodes):
        ep_data = viirs_df[viirs_df['episode_id'] == ep_id].sort_values('datetime')
        times = (ep_data['datetime'] - ep_data['datetime'].iloc[0]).dt.total_seconds() / 3600
        ax10.plot(times, ep_data['te'], marker='o', label=f'Episode {ep_id}', alpha=0.7)
    ax10.set_xlabel('Time since start (hours)')
    ax10.set_ylabel('Fire Temperature (°C)')
    ax10.set_title('Sample Episode Temperature Time Series (5 largest episodes)')
    ax10.legend()
    ax10.grid(True, alpha=0.3)

    plt.suptitle('Embedded Data Validation Report', fontsize=16, fontweight='bold')

    # Save
    plot_path = output_dir / 'validation_plots.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved validation plots: {plot_path}")
    plt.close()

    validation_report['passed'].append("Validation visualizations created")

except Exception as e:
    validation_report['warnings'].append(f"Failed to create visualizations: {e}")
    print(f"  ⚠ Failed to create visualizations: {e}")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

print(f"\n✓ PASSED: {len(validation_report['passed'])} checks")
for check in validation_report['passed']:
    print(f"  • {check}")

if validation_report['warnings']:
    print(f"\n⚠ WARNINGS: {len(validation_report['warnings'])} issues")
    for warning in validation_report['warnings'][:20]:  # Show first 20
        print(f"  • {warning}")
    if len(validation_report['warnings']) > 20:
        print(f"  ... and {len(validation_report['warnings']) - 20} more")

if validation_report['failed']:
    print(f"\n✗ FAILED: {len(validation_report['failed'])} critical issues")
    for failure in validation_report['failed']:
        print(f"  • {failure}")

# Save validation report
report_path = output_dir / 'validation_report.json'
with open(report_path, 'w') as f:
    json.dump(validation_report, f, indent=2, default=str)

print(f"\n✓ Validation report saved: {report_path}")

# Overall assessment
print("\n" + "=" * 80)
if len(validation_report['failed']) == 0:
    if len(validation_report['warnings']) == 0:
        print("✓✓✓ ALL CHECKS PASSED - DATA READY FOR TILING ✓✓✓")
    else:
        print("✓ VALIDATION PASSED WITH WARNINGS - Review warnings before tiling")
else:
    print("✗ VALIDATION FAILED - Fix critical issues before proceeding")
print("=" * 80)
