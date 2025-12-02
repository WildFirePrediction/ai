"""
Test script for production inference components
Tests each component independently before running full loop
"""
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from inference.sl.grid_utils import (
    create_fire_grid,
    create_initial_fire_mask,
    grid_cell_to_latlon,
    latlon_to_epsg5179,
    epsg5179_to_latlon
)
from inference.sl.data_pipeline import (
    StaticDataLoader,
    fetch_kma_weather,
    process_weather_data,
    create_input_tensor
)
from inference.sl.inference_engine import WildfireInferenceEngine


def test_coordinate_conversion():
    """Test coordinate conversion"""
    print("\n" + "=" * 80)
    print("TEST 1: Coordinate Conversion")
    print("=" * 80)

    # Test with known location (Seoul City Hall)
    lat, lon = 37.5665, 126.9780
    print(f"Input (WGS84): lat={lat}, lon={lon}")

    # Convert to EPSG:5179
    x, y = latlon_to_epsg5179(lat, lon)
    print(f"EPSG:5179: x={x:.2f}m, y={y:.2f}m")

    # Convert back
    lat2, lon2 = epsg5179_to_latlon(x, y)
    print(f"Back to WGS84: lat={lat2}, lon={lon2}")

    # Check accuracy
    error = np.sqrt((lat - lat2)**2 + (lon - lon2)**2)
    print(f"Round-trip error: {error:.10f} degrees")

    if error < 1e-6:
        print("PASS: Coordinate conversion accurate")
    else:
        print("FAIL: Coordinate conversion has errors")

    return error < 1e-6


def test_grid_creation():
    """Test grid creation"""
    print("\n" + "=" * 80)
    print("TEST 2: Grid Creation")
    print("=" * 80)

    # Use current fire location from KFS
    lat, lon = 35.4396, 127.2951
    print(f"Fire location: ({lat}, {lon})")

    # Create grid
    grid_bounds, grid_coords, center_xy = create_fire_grid(lat, lon)
    print(f"Grid bounds: {grid_bounds}")
    print(f"Grid coords shape: {grid_coords.shape}")
    print(f"Center XY: {center_xy}")

    # Check grid is 30x30
    assert grid_coords.shape == (30, 30, 2), "Grid should be 30x30x2"

    # Create fire mask
    fire_mask = create_initial_fire_mask(center_xy, grid_coords)
    print(f"Fire mask shape: {fire_mask.shape}")
    print(f"Fire cells: {(fire_mask > 0).sum()}")

    # Check fire mask has exactly 1 cell
    assert (fire_mask > 0).sum() == 1, "Fire mask should have 1 cell"

    # Convert center cell back to lat/lon
    fire_cell = np.where(fire_mask > 0)
    fire_row, fire_col = fire_cell[0][0], fire_cell[1][0]
    lat_back, lon_back = grid_cell_to_latlon(fire_row, fire_col, grid_coords)
    print(f"Fire cell: ({fire_row}, {fire_col})")
    print(f"Fire cell lat/lon: ({lat_back:.4f}, {lon_back:.4f})")

    # Check distance from original
    distance = np.sqrt((lat - lat_back)**2 + (lon - lon_back)**2) * 111000  # rough meters
    print(f"Distance from original: {distance:.0f}m")

    if distance < 500:  # Should be within one cell (400m)
        print("PASS: Grid creation accurate")
        return True
    else:
        print("FAIL: Grid creation has large error")
        return False


def test_static_data():
    """Test static data loading"""
    print("\n" + "=" * 80)
    print("TEST 3: Static Data Loading")
    print("=" * 80)

    try:
        loader = StaticDataLoader()
        print("Static data loader initialized")

        # Create test grid
        lat, lon = 35.4396, 127.2951
        grid_bounds, _, _ = create_fire_grid(lat, lon)

        # Extract static features
        static_channels = loader.extract_static_features(grid_bounds)
        print(f"Static channels shape: {static_channels.shape}")

        # Check shape
        assert static_channels.shape == (7, 30, 30), "Static channels should be (7, 30, 30)"

        # Check for non-zero values
        for i in range(7):
            non_zero = (static_channels[i] != 0).sum()
            print(f"  Channel {i}: {non_zero}/900 non-zero cells")

        print("PASS: Static data loading works")
        return True

    except Exception as e:
        print(f"FAIL: Static data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_weather_fetching():
    """Test weather data fetching"""
    print("\n" + "=" * 80)
    print("TEST 4: Weather Data Fetching")
    print("=" * 80)

    # Use a recent timestamp
    timestamp = datetime(2025, 11, 29, 16, 30)
    kma_url = 'https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min?authKey=ud6z-X7yRDKes_l-8qQyFg'

    print(f"Fetching weather for {timestamp}...")
    df_weather = fetch_kma_weather(timestamp, kma_url)

    if df_weather is not None and len(df_weather) > 0:
        print(f"Success: Found {len(df_weather)} weather stations")
        print(f"Columns: {df_weather.columns.tolist()}")
        print("\nSample data:")
        print(df_weather[['STN', 'TA', 'HM', 'WS1', 'PA']].head())
        print("PASS: Weather fetching works")
        return True
    else:
        print("FAIL: No weather data fetched")
        return False


def test_weather_processing():
    """Test weather processing"""
    print("\n" + "=" * 80)
    print("TEST 5: Weather Processing")
    print("=" * 80)

    # Fetch weather
    timestamp = datetime(2025, 11, 29, 16, 30)
    kma_url = 'https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min?authKey=ud6z-X7yRDKes_l-8qQyFg'
    df_weather = fetch_kma_weather(timestamp, kma_url)

    # Create grid
    lat, lon = 35.4396, 127.2951
    _, _, center_xy = create_fire_grid(lat, lon)

    # Process weather
    weather_channels = process_weather_data(df_weather, center_xy)
    print(f"Weather channels shape: {weather_channels.shape}")

    # Check shape
    assert weather_channels.shape == (9, 30, 30), "Weather channels should be (9, 30, 30)"

    # Check channels 8, 9 are zero (cloud, visibility)
    assert np.all(weather_channels[6] == 0), "Channel 6 (cloud) should be zero"
    assert np.all(weather_channels[7] == 0), "Channel 7 (visibility) should be zero"

    # Check other channels have values
    print("Weather channel statistics:")
    for i in range(9):
        print(f"  Channel {i}: min={weather_channels[i].min():.4f}, "
              f"max={weather_channels[i].max():.4f}, "
              f"mean={weather_channels[i].mean():.4f}")

    print("PASS: Weather processing works")
    return True


def test_input_tensor():
    """Test input tensor creation"""
    print("\n" + "=" * 80)
    print("TEST 6: Input Tensor Creation")
    print("=" * 80)

    # Load static data
    loader = StaticDataLoader()
    lat, lon = 35.4396, 127.2951
    grid_bounds, grid_coords, center_xy = create_fire_grid(lat, lon)
    static_channels = loader.extract_static_features(grid_bounds)

    # Create dummy weather
    weather_channels = np.random.rand(9, 30, 30).astype(np.float32)

    # Create fire mask
    fire_mask = create_initial_fire_mask(center_xy, grid_coords)

    # Create input tensor
    input_data = create_input_tensor(static_channels, weather_channels, fire_mask)
    print(f"Input tensor shape: {input_data.shape}")

    # Check shape
    assert input_data.shape == (17, 30, 30), "Input should be (17, 30, 30)"

    # Check channel order
    print("Channel check:")
    print(f"  Ch 0-1 (slope, aspect): unique values = {np.unique(input_data[0:2]).size}")
    print(f"  Ch 2-10 (weather): min={input_data[2:11].min():.4f}, max={input_data[2:11].max():.4f}")
    print(f"  Ch 11 (NDVI): unique values = {np.unique(input_data[11]).size}")
    print(f"  Ch 12-15 (FSM): unique values = {np.unique(input_data[12:16]).size}")
    print(f"  Ch 16 (fire): sum = {input_data[16].sum()}")

    print("PASS: Input tensor creation works")
    return True


def test_inference():
    """Test inference engine"""
    print("\n" + "=" * 80)
    print("TEST 7: Inference Engine")
    print("=" * 80)

    checkpoint_path = 'sl_training/unet_16ch_v3/checkpoints/run1_dilated(0.3642)/best.pt'

    try:
        engine = WildfireInferenceEngine(checkpoint_path, device='cuda')
        print("Inference engine loaded")

        # Create dummy input
        input_data = np.random.rand(17, 30, 30).astype(np.float32)

        # Run inference
        predictions = engine.predict(input_data)
        print(f"Predictions shape: {predictions.shape}")

        # Check output shape
        assert predictions.shape == (3, 30, 30), "Predictions should be (3, 30, 30)"

        # Check values are probabilities
        assert predictions.min() >= 0 and predictions.max() <= 1, "Predictions should be in [0, 1]"

        print(f"Prediction statistics:")
        for t in range(3):
            cells_above_05 = (predictions[t] > 0.5).sum()
            print(f"  t+{t+1}: min={predictions[t].min():.4f}, "
                  f"max={predictions[t].max():.4f}, "
                  f"cells>0.5={cells_above_05}")

        print("PASS: Inference engine works")
        return True

    except Exception as e:
        print(f"FAIL: Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("PRODUCTION INFERENCE - COMPONENT TESTS")
    print("=" * 80)

    results = {}

    # Run tests
    results['coordinate_conversion'] = test_coordinate_conversion()
    results['grid_creation'] = test_grid_creation()
    results['static_data'] = test_static_data()
    results['weather_fetching'] = test_weather_fetching()
    results['weather_processing'] = test_weather_processing()
    results['input_tensor'] = test_input_tensor()
    results['inference'] = test_inference()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:30s}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\nAll tests passed! Ready for production.")
    else:
        print("\nSome tests failed. Fix issues before running production.")


if __name__ == '__main__':
    main()
