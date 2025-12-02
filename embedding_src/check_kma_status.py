#!/usr/bin/env python3
"""
Quick check of KMA data status
"""
from pathlib import Path

print("=" * 80)
print("KMA DATA STATUS CHECK")
print("=" * 80)

# Get absolute paths
script_dir = Path(__file__).parent
root_dir = script_dir.parent

# Count timestamps needed
timestamps_file = root_dir / 'embedded_data' / 'kma_timestamps_to_fetch.txt'
if timestamps_file.exists():
    with open(timestamps_file, 'r') as f:
        timestamps_needed = [line.strip() for line in f if line.strip()]
    print(f"\n📋 Timestamps needed: {len(timestamps_needed)}")
    print(f"   First: {timestamps_needed[0]}")
    print(f"   Last: {timestamps_needed[-1]}")
else:
    print(f"\n❌ No timestamps file found: {timestamps_file}")
    timestamps_needed = []

# Count what we have
kma_dir = root_dir / 'data' / 'KMA'
if kma_dir.exists():
    kma_subdirs = [d for d in kma_dir.iterdir() if d.is_dir() and d.name.startswith('20')]
    print(f"\n📁 KMA directories found: {len(kma_subdirs)}")

    # Count CSV files
    csv_files = list(kma_dir.rglob('AWS_*.csv'))
    print(f"📄 AWS CSV files found: {len(csv_files)}")

    if kma_subdirs:
        print(f"\n   Examples:")
        for d in sorted(kma_subdirs)[:5]:
            csv_count = len(list(d.glob('AWS_*.csv')))
            print(f"     {d.name}: {csv_count} CSV files")

    # Check coverage
    existing_timestamps = set(d.name for d in kma_subdirs)
    needed_timestamps = set(timestamps_needed)
    missing = needed_timestamps - existing_timestamps

    print(f"\n✅ Already downloaded: {len(existing_timestamps)}")
    print(f"⏳ Still missing: {len(missing)}")

    if len(missing) > 0:
        print(f"\n   Missing examples: {sorted(list(missing))[:10]}")

else:
    print(f"\n❌ KMA data directory not found: {kma_dir}")

print("\n" + "=" * 80)

