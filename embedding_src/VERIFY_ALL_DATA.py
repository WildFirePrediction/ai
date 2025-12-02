"""
Comprehensive Data Integrity Verification
Checks DEM, KMA weather data, and VIIRS fire data
"""

import os
import numpy as np
import rasterio
from pathlib import Path
import json

root_dir = Path(__file__).parent.parent
data_dir = root_dir / 'data'
embedded_dir = root_dir / 'embedded_data'

print("=" * 80)
print("COMPREHENSIVE DATA INTEGRITY CHECK")
print("=" * 80)

# ============================================================================
# 1. DEM DATA
# ============================================================================
print("\n[1/4] DEM Data...")

dem_tif = embedded_dir / 'dem_slope_aspect_native_crs.tif'
if dem_tif.exists():
    with rasterio.open(dem_tif) as src:
        print(f"  File: {dem_tif.name}")
        print(f"  Shape: {src.shape}")
        print(f"  Channels: {src.count}")
        print(f"  CRS: {src.crs}")
        
        for i in range(1, src.count + 1):
            data = src.read(i)
            valid = data[~np.isnan(data)]
            pct = len(valid) / data.size * 100
            print(f"  Channel {i}: {len(valid):,}/{data.size:,} valid ({pct:.2f}%)")
            if len(valid) > 0:
                print(f"    Range: [{np.min(valid):.2f}, {np.max(valid):.2f}]")
else:
    print(f"  ERROR: {dem_tif} not found!")

# ============================================================================
# 2. KMA WEATHER DATA FILES
# ============================================================================
print("\n[2/4] KMA Raw Data Files...")

kma_dir = data_dir / 'KMA'
if kma_dir.exists():
    csv_files = list(kma_dir.glob('*.csv'))
    print(f"  KMA directory: {kma_dir}")
    print(f"  CSV files found: {len(csv_files)}")
    
    if len(csv_files) > 0:
        # Sample first file
        first_file = csv_files[0]
        print(f"  Sample file: {first_file.name}")
        with open(first_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"    Lines: {len(lines)}")
            if len(lines) > 0:
                print(f"    Header: {lines[0].strip()}")
            if len(lines) > 1:
                print(f"    First data: {lines[1].strip()}")
                
        # Check date range
        dates = set()
        for f in csv_files[:100]:  # Sample first 100
            fname = f.stem  # e.g., '202301010000'
            if len(fname) >= 8:
                dates.add(fname[:8])  # YYYYMMDD
        
        if dates:
            dates_sorted = sorted(dates)
            print(f"  Date range (sampled): {dates_sorted[0]} to {dates_sorted[-1]}")
            print(f"  Unique dates: {len(dates_sorted)}")
    else:
        print("  WARNING: No CSV files found in KMA directory")
else:
    print(f"  ERROR: {kma_dir} not found!")

# ============================================================================
# 3. KMA EMBEDDED WEATHER DATA
# ============================================================================
print("\n[3/4] KMA Embedded Weather...")

kma_weather_dir = embedded_dir / 'kma_weather'
if kma_weather_dir.exists():
    tif_files = list(kma_weather_dir.glob('weather_*.tif'))
    print(f"  Embedded weather files: {len(tif_files)}")
    
    if len(tif_files) > 0:
        # Check first file
        first_tif = tif_files[0]
        with rasterio.open(first_tif) as src:
            print(f"  Sample: {first_tif.name}")
            print(f"    Shape: {src.shape}")
            print(f"    Channels: {src.count}")
            print(f"    CRS: {src.crs}")
            
            channel_names = ['temp', 'humidity', 'wind_u', 'wind_v', 'wind_speed', 
                           'precip', 'pressure', 'cloud_index', 'visibility']
            
            for i in range(1, min(src.count + 1, 10)):
                data = src.read(i)
                valid = data[~np.isnan(data)]
                pct = len(valid) / data.size * 100
                cname = channel_names[i-1] if i <= len(channel_names) else f'ch{i}'
                print(f"    {cname}: {len(valid):,}/{data.size:,} valid ({pct:.2f}%)")
                if len(valid) > 0:
                    print(f"      Range: [{np.min(valid):.2f}, {np.max(valid):.2f}]")
    else:
        print("  INFO: No embedded weather files yet (expected before full embedding)")
else:
    print(f"  INFO: {kma_weather_dir} doesn't exist yet")

# ============================================================================
# 4. VIIRS FIRE DATA
# ============================================================================
print("\n[4/4] VIIRS Fire Data...")

viirs_parquet = data_dir / 'filtered_fires' / 'filtered_viirs.parquet'
if viirs_parquet.exists():
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(viirs_parquet)
        df = table.to_pandas()
        print(f"  File: {viirs_parquet.name}")
        print(f"  Records: {len(df)}")
        print(f"  Columns: {', '.join(df.columns)}")
        
        # Check for required columns
        date_col = 'acq_date' if 'acq_date' in df.columns else 'ACQ_DATE' if 'ACQ_DATE' in df.columns else None
        time_col = 'acq_time' if 'acq_time' in df.columns else 'ACQ_TIME' if 'ACQ_TIME' in df.columns else None
        
        if date_col and time_col:
            print(f"  Date column: {date_col}")
            print(f"  Time column: {time_col}")
            print(f"  Date range: {df[date_col].min()} to {df[date_col].max()}")
            print(f"  Unique dates: {df[date_col].nunique()}")
        else:
            print(f"  WARNING: Missing date/time columns!")
            print(f"  Available: {', '.join(df.columns)}")
            
    except Exception as e:
        print(f"  ERROR reading parquet: {e}")
else:
    print(f"  ERROR: {viirs_parquet} not found!")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

checks = {
    'DEM embedded': dem_tif.exists(),
    'KMA raw data': kma_dir.exists() and len(list(kma_dir.glob('*.csv'))) > 0,
    'VIIRS fire data': viirs_parquet.exists(),
}

for check, status in checks.items():
    symbol = "✓" if status else "✗"
    print(f"{symbol} {check}")

print("=" * 80)
