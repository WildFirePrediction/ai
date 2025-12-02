"""
Dummy KFS API data for testing inference pipeline
Provides realistic fire trigger data from past fire events in training dataset
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

import pickle
import numpy as np
from datetime import datetime, timedelta

def get_test_fire_location():
    """
    Get a fire location with verified terrain data coverage.
    Uses pre-validated locations known to have good DEM/NDVI variation.
    """
    # Use pre-validated locations with confirmed terrain data
    # These locations were tested and have DEM std > 0.05 and NDVI std > 0.1
    
    good_locations = [
        # (lat, lon, region_name)
        (36.67, 127.89, "Central South Korea (Gyeongnam)"),
        (37.03, 128.57, "Gangwon-do Mountains"),
        (39.38, 130.82, "Far East (near coast)"),
    ]
    
    # Pick a random location from validated ones
    base_lat, base_lon, region = good_locations[np.random.randint(len(good_locations))]
    
    # Add small random offset (±0.1 degrees, ~10km)
    lat = base_lat + np.random.uniform(-0.1, 0.1)
    lon = base_lon + np.random.uniform(-0.1, 0.1)
    
    # Convert lat/lon to EPSG:5179 (rough approximation)
    # For testing - production should use pyproj
    x_epsg5179 = (lon - 127.5) * 88800 + 1000000
    y_epsg5179 = (lat - 38.0) * 111000 + 1800000
    
    # Grid metadata for coordinate conversion
    grid_meta_path = Path(__file__).parent.parent.parent / 'embedded_data' / 'grid_metadata.json'
    import json
    with open(grid_meta_path, 'r') as f:
        grid_meta = json.load(f)
    
    transform = grid_meta['transform']  # [a, b, c, d, e, f]
    a, b, c, d, e, f = transform
    
    # Convert to grid coordinates
    global_col = int(round((x_epsg5179 - c) / a))
    global_row = int(round((f - y_epsg5179) / abs(e)))
    
    return {
        'lat': lat,
        'lon': lon,
        'x_epsg5179': x_epsg5179,
        'y_epsg5179': y_epsg5179,
        'global_row': global_row,
        'global_col': global_col,
        'region': region,
        'note': 'Verified location with terrain data coverage'
    }


def generate_dummy_kfs_trigger():
    """
    Generate a dummy KFS fire trigger for testing.
    Mimics the structure of kfs_api.py - get_kfs_fire_data()
    
    Returns:
        dict: Fire trigger data with lat, lon, timestamp
    """
    fire_loc = get_test_fire_location()
    
    # Generate timestamp (current time for testing)
    timestamp = datetime.now()
    
    return {
        'lat': fire_loc['lat'],
        'lon': fire_loc['lon'],
        'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        'timestamp_iso': timestamp.isoformat(),
        'fire_id': 'TEST_FIRE_001',
        'location_name': f"Test Fire - {fire_loc['region']}",
        # Debug info (not in real KFS data)
        '_debug': {
            'x_epsg5179': fire_loc['x_epsg5179'],
            'y_epsg5179': fire_loc['y_epsg5179'],
            'global_row': fire_loc['global_row'],
            'global_col': fire_loc['global_col'],
            'region': fire_loc['region'],
            'note': fire_loc['note']
        }
    }


def generate_progressive_fire_triggers(num_triggers=3, interval_hours=2):
    """
    Generate multiple fire triggers simulating fire spread over time.
    Useful for testing hard-reset mechanism.
    
    Args:
        num_triggers: Number of triggers to generate
        interval_hours: Hours between triggers
        
    Returns:
        list: List of fire trigger dicts
    """
    base_trigger = generate_dummy_kfs_trigger()
    triggers = [base_trigger]
    
    base_time = datetime.fromisoformat(base_trigger['timestamp_iso'])
    
    for i in range(1, num_triggers):
        new_time = base_time + timedelta(hours=interval_hours * i)
        
        # Simulate fire spreading (slightly different location)
        lat_offset = np.random.uniform(-0.01, 0.01)
        lon_offset = np.random.uniform(-0.01, 0.01)
        
        new_trigger = {
            'lat': base_trigger['lat'] + lat_offset,
            'lon': base_trigger['lon'] + lon_offset,
            'timestamp': new_time.strftime("%Y-%m-%d %H:%M:%S"),
            'timestamp_iso': new_time.isoformat(),
            'fire_id': f'TEST_FIRE_{i+1:03d}',
            'location_name': f'Test Fire Location {i+1} (spread)',
        }
        triggers.append(new_trigger)
    
    return triggers


if __name__ == '__main__':
    # Test dummy data generation
    print("=" * 80)
    print("DUMMY KFS FIRE TRIGGER TEST")
    print("=" * 80)
    
    trigger = generate_dummy_kfs_trigger()
    print("\nSingle trigger:")
    print(f"  Fire ID: {trigger['fire_id']}")
    print(f"  Location: {trigger['lat']:.6f}N, {trigger['lon']:.6f}E")
    print(f"  Timestamp: {trigger['timestamp']}")
    print(f"  Debug info:")
    print(f"    EPSG:5179: ({trigger['_debug']['x_epsg5179']:.1f}, {trigger['_debug']['y_epsg5179']:.1f})")
    print(f"    Global grid: (row={trigger['_debug']['global_row']}, col={trigger['_debug']['global_col']})")
    print(f"    Region: {trigger['_debug']['region']}")
    print(f"    Note: {trigger['_debug']['note']}")
    
    print("\n" + "=" * 80)
    print("PROGRESSIVE TRIGGERS TEST (Hard Reset Simulation)")
    print("=" * 80)
    
    triggers = generate_progressive_fire_triggers(num_triggers=3, interval_hours=2)
    for i, trig in enumerate(triggers):
        print(f"\nTrigger {i+1}:")
        print(f"  Location: {trig['lat']:.6f}N, {trig['lon']:.6f}E")
        print(f"  Time: {trig['timestamp']}")
