#!/usr/bin/env python3
"""
Fetch KMA weather data for all NASA VIIRS fire detection timestamps
"""
import sys
import time
from pathlib import Path

script_dir = Path(__file__).parent
root_dir = script_dir.parent
src_path = root_dir / 'src'
sys.path.insert(0, str(src_path))

from kma_api import get_all_weather_data

def main():
    # Get absolute paths
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent

    # Read timestamps
    timestamps_file = root_dir / 'embedded_data' / 'kma_timestamps_to_fetch.txt'

    if not timestamps_file.exists():
        print(f"ERROR: {timestamps_file} not found!")
        print("Run analyze_viirs_temporal.py first to generate timestamps")
        return

    with open(timestamps_file, 'r') as f:
        timestamps = [line.strip() for line in f if line.strip()]

    print("=" * 80)
    print("KMA WEATHER DATA BATCH FETCH")
    print("=" * 80)
    print(f"\n📋 Total timestamps to fetch: {len(timestamps)}")
    print(f"📅 Date range: {timestamps[0]} to {timestamps[-1]}")

    # Check which ones already exist
    kma_data_dir = root_dir / 'data' / 'KMA'
    existing = []
    missing = []

    for ts in timestamps:
        out_dir = kma_data_dir / ts
        if out_dir.exists() and any(out_dir.glob('AWS_*.csv')):
            existing.append(ts)
        else:
            missing.append(ts)

    print(f"\n✅ Already downloaded: {len(existing)}")
    print(f"⏳ Need to download: {len(missing)}")

    if not missing:
        print("\n🎉 All KMA data already downloaded!")
        return

    # Ask for confirmation
    print(f"\n⚠️  This will fetch {len(missing)} timestamps from KMA API")
    print(f"⚠️  Estimated time: ~{len(missing) * 2} seconds ({len(missing) * 2 / 60:.1f} minutes)")

    response = input("\nProceed with download? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return

    # Fetch missing data
    print(f"\n🚀 Starting download...")
    print(f"📊 Progress:")

    success_count = 0
    error_count = 0

    for i, timestamp in enumerate(missing, 1):
        try:
            print(f"  [{i}/{len(missing)}] Fetching {timestamp}...", end=" ", flush=True)
            get_all_weather_data(timestamp)
            success_count += 1
            print("✓")

            # Rate limiting - be nice to the API
            time.sleep(1)

        except Exception as e:
            print(f"✗ Error: {e}")
            error_count += 1
            continue

    print("\n" + "=" * 80)
    print("FETCH COMPLETE")
    print("=" * 80)
    print(f"✅ Success: {success_count}")
    print(f"❌ Errors: {error_count}")
    print(f"📁 Data saved to: {kma_data_dir.absolute()}")
    print("=" * 80)

if __name__ == "__main__":
    main()

