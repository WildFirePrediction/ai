"""
Data pipeline for real-time wildfire inference
Handles static data extraction and weather data fetching
"""
import numpy as np
import rasterio
from pathlib import Path
import requests
import pandas as pd
from datetime import datetime
import sys
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from inference.sl.grid_utils import latlon_to_raster_crs


# Weather normalization parameters (from training)
WEATHER_NORM = {
    'temperature': {'min': -30, 'max': 40},
    'humidity': {'min': 0, 'max': 100},
    'wind_speed': {'min': 0, 'max': 30},
    'wind_direction': {'min': 0, 'max': 360},
    'precipitation': {'min': 0, 'max': 50},
    'pressure': {'min': 950, 'max': 1050},
    'dew_point': {'min': -40, 'max': 30}
}

# KMA weather station coordinates (all 78 major stations)
KMA_STATIONS = {
    90: (38.25, 128.564),
    93: (37.805, 128.859),
    95: (38.147, 127.308),
    98: (37.903, 127.061),
    99: (37.908, 126.78),
    100: (37.677, 128.718),
    101: (37.902, 127.736),
    102: (37.974, 124.63),
    104: (37.95, 127.745),
    105: (37.751, 128.891),
    106: (37.501, 129.128),
    108: (37.571, 126.966),
    112: (37.478, 126.625),
    114: (37.338, 127.946),
    115: (37.481, 130.9),
    119: (37.273, 126.987),
    121: (37.183, 128.461),
    127: (36.97, 127.953),
    129: (36.774, 126.495),
    130: (36.991, 129.407),
    131: (36.638, 127.44),
    133: (36.369, 127.374),
    135: (36.218, 127.994),
    136: (36.573, 128.707),
    137: (36.411, 128.159),
    138: (36.033, 129.38),
    140: (35.984, 126.563),
    143: (35.885, 128.652),
    146: (35.821, 127.155),
    152: (35.56, 129.322),
    155: (35.18, 128.55),
    156: (35.172, 126.891),
    159: (35.104, 129.032),
    162: (34.845, 128.435),
    165: (34.817, 126.381),
    168: (34.739, 127.74),
    170: (34.4, 126.702),
    172: (34.618, 127.275),
    174: (33.514, 126.53),
    177: (33.292, 126.163),
    184: (33.246, 126.565),
    185: (35.192, 128.043),
    188: (37.707, 126.445),
    189: (37.488, 127.495),
    192: (37.266, 127.47),
    201: (38.06, 128.167),
    202: (37.683, 127.883),
    203: (37.164, 128.986),
    211: (37.156, 128.197),
    212: (36.487, 127.733),
    216: (36.777, 127.12),
    217: (36.333, 126.556),
    221: (36.272, 126.921),
    226: (36.105, 127.488),
    232: (35.61, 127.285),
    235: (35.563, 126.866),
    236: (35.416, 127.39),
    238: (34.689, 126.919),
    239: (34.553, 126.569),
    243: (36.355, 128.688),
    244: (36.133, 128.319),
    245: (35.977, 128.951),
    246: (35.843, 129.223),
    247: (35.671, 127.912),
    248: (35.566, 128.166),
    251: (35.493, 128.738),
    252: (35.416, 127.873),
    253: (34.89, 128.605),
    254: (34.8, 127.926),
    255: (33.523, 126.897),
    256: (33.285, 126.162),
    257: (33.387, 126.88),
    258: (36.783, 127.003),
    259: (36.871, 128.517),
    260: (36.627, 128.147),
    261: (36.533, 129.408),
    262: (36.362, 128.683),
    263: (36.138, 128.327),
}


class StaticDataLoader:
    """Loads and caches static environmental rasters"""

    def __init__(self, data_dir='embedded_data'):
        self.data_dir = Path(data_dir)
        self.dem_raster = None
        self.ndvi_raster = None
        self.fsm_raster = None
        self._load_rasters()

    def _load_rasters(self):
        """Load all static rasters (cached)"""
        dem_path = self.data_dir / 'dem_slope_aspect_FINAL.tif'
        ndvi_path = self.data_dir / 'ndvi_FINAL.tif'
        fsm_path = self.data_dir / 'fsm_FINAL.tif'

        if not dem_path.exists():
            raise FileNotFoundError(f"DEM raster not found: {dem_path}")
        if not ndvi_path.exists():
            raise FileNotFoundError(f"NDVI raster not found: {ndvi_path}")
        if not fsm_path.exists():
            raise FileNotFoundError(f"FSM raster not found: {fsm_path}")

        self.dem_raster = rasterio.open(dem_path)
        self.ndvi_raster = rasterio.open(ndvi_path)
        self.fsm_raster = rasterio.open(fsm_path)

        print(f"Static rasters loaded:")
        print(f"  DEM: {dem_path}")
        print(f"  NDVI: {ndvi_path}")
        print(f"  FSM: {fsm_path}")

    def extract_static_features(self, grid_bounds):
        """
        Extract static features for grid region

        Args:
            grid_bounds: (x_min, x_max, y_min, y_max) in EPSG:5179

        Returns:
            static_channels: (7, 30, 30) array
                - Channel 0: Slope
                - Channel 1: Aspect
                - Channel 2: NDVI
                - Channels 3-6: FSM one-hot (4 classes)
        """
        x_min, x_max, y_min, y_max = grid_bounds

        # Extract window from each raster
        # DEM has 3 bands: elevation, slope, aspect (we need bands 2, 3)
        dem_window = rasterio.windows.from_bounds(
            x_min, y_min, x_max, y_max,
            transform=self.dem_raster.transform
        )
        slope = self.dem_raster.read(2, window=dem_window)  # Band 2: slope
        aspect = self.dem_raster.read(3, window=dem_window)  # Band 3: aspect

        # NDVI (single band)
        ndvi_window = rasterio.windows.from_bounds(
            x_min, y_min, x_max, y_max,
            transform=self.ndvi_raster.transform
        )
        ndvi = self.ndvi_raster.read(1, window=ndvi_window)

        # FSM (single band with values 1-4)
        fsm_window = rasterio.windows.from_bounds(
            x_min, y_min, x_max, y_max,
            transform=self.fsm_raster.transform
        )
        fsm = self.fsm_raster.read(1, window=fsm_window)

        # Resize to exactly 30x30 if needed
        from scipy.ndimage import zoom
        target_shape = (30, 30)

        # Check if extraction succeeded
        if slope.size == 0 or slope.shape[0] == 0 or slope.shape[1] == 0:
            print(f"WARNING: Empty raster window, filling with default values")
            slope = np.zeros(target_shape, dtype=np.float32)
            aspect = np.zeros(target_shape, dtype=np.float32)
            ndvi = np.zeros(target_shape, dtype=np.float32)
            fsm = np.ones(target_shape, dtype=np.float32)  # Default FSM class 1
        elif slope.shape != target_shape:
            zoom_factor = (target_shape[0] / slope.shape[0],
                          target_shape[1] / slope.shape[1])
            slope = zoom(slope, zoom_factor, order=1)
            aspect = zoom(aspect, zoom_factor, order=1)
            ndvi = zoom(ndvi, zoom_factor, order=1)
            fsm = zoom(fsm, zoom_factor, order=0)  # Nearest neighbor for categorical

        # One-hot encode FSM (4 classes: 1-4)
        fsm_onehot = np.zeros((4, 30, 30), dtype=np.float32)
        for class_idx in range(1, 5):
            fsm_onehot[class_idx - 1] = (fsm == class_idx).astype(np.float32)

        # Stack: slope, aspect, ndvi, fsm_onehot
        static_channels = np.concatenate([
            slope[np.newaxis, :, :],   # (1, 30, 30)
            aspect[np.newaxis, :, :],  # (1, 30, 30)
            ndvi[np.newaxis, :, :],    # (1, 30, 30)
            fsm_onehot                 # (4, 30, 30)
        ], axis=0)  # (7, 30, 30)

        return static_channels


def fetch_kma_weather(timestamp, kma_api_url):
    """
    Fetch weather data from KMA API with retry logic

    Args:
        timestamp: datetime object or "YYYYMMDDHHMM" string
        kma_api_url: KMA API URL with auth key

    Returns:
        df_weather: DataFrame with columns [STN, TA, HM, WS1, WD1, RN-15m, PA, TD]
                    or None if all retries fail
    """
    if isinstance(timestamp, datetime):
        timestamp_str = timestamp.strftime("%Y%m%d%H%M")
    else:
        timestamp_str = str(timestamp)

    url = f"{kma_api_url}&tm2={timestamp_str}"

    max_retries = 10
    retry_interval = 15  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            if attempt > 1:
                print(f"[KMA API] Retry attempt {attempt}/{max_retries}...")

            response = requests.get(url, timeout=15)
            response.encoding = 'euc-kr'

            # Parse CSV (skip comment lines starting with #)
            lines = response.text.split('\n')

            # Find header line
            header_line = None
            data_start_idx = None
            for i, line in enumerate(lines):
                if line.startswith('#') and 'YYMMDDHHMI' in line:
                    header_line = line.lstrip('#').strip()
                elif not line.startswith('#') and line.strip() and header_line:
                    data_start_idx = i
                    break

            if not header_line or data_start_idx is None:
                print(f"Failed to parse KMA response (no valid data)")
                if attempt < max_retries:
                    print(f"Retrying in {retry_interval}s...")
                    time.sleep(retry_interval)
                    continue
                return None

            # Clean header
            headers = [h.rstrip('.') for h in header_line.split()]
            if headers[0].upper().startswith('YYMMDD'):
                headers[0] = 'TIME'

            # Parse data rows
            data_rows = []
            for line in lines[data_start_idx:]:
                if not line.strip() or line.startswith('#'):
                    continue
                parts = line.strip().split()
                data_rows.append(parts)

            if not data_rows:
                print(f"No data rows found in KMA response")
                if attempt < max_retries:
                    print(f"Retrying in {retry_interval}s...")
                    time.sleep(retry_interval)
                    continue
                return None

            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=headers)

            # Convert numeric columns
            numeric_cols = ['STN', 'TA', 'HM', 'WS1', 'WD1', 'RN-15m', 'PA', 'TD']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Replace missing values (-99, -99.9, -999) with NaN
            df = df.replace([-99, -99.9, -999], np.nan)

            if attempt > 1:
                print(f"[KMA API] Success after {attempt} attempts")

            return df

        except requests.exceptions.Timeout:
            print(f"[KMA API] Timeout on attempt {attempt}/{max_retries}")
            if attempt < max_retries:
                print(f"Retrying in {retry_interval}s...")
                time.sleep(retry_interval)
            else:
                print(f"[KMA API] All {max_retries} attempts failed")
                return None
        except Exception as e:
            print(f"Error fetching KMA weather: {e}")
            if attempt < max_retries:
                print(f"Retrying in {retry_interval}s...")
                time.sleep(retry_interval)
            else:
                print(f"[KMA API] All {max_retries} attempts failed")
                return None

    return None


def process_weather_data(df_weather, center_xy, grid_size=30):
    """
    Process weather data: find closest station and create grid

    Args:
        df_weather: DataFrame from fetch_kma_weather()
        center_xy: (center_x, center_y) fire location in raster CRS
        grid_size: Grid dimensions (default 30)

    Returns:
        weather_channels: (9, 30, 30) array with weather data
                         Channels: temp, humid, wind_speed, wind_dir,
                                  precip, pressure, cloud(0), visibility(0), dew_point
    """
    weather_channels = np.zeros((9, grid_size, grid_size), dtype=np.float32)

    if df_weather is None or len(df_weather) == 0:
        print("No weather data available, using zeros")
        return weather_channels

    center_x, center_y = center_xy

    # Find closest weather station
    min_distance = float('inf')
    closest_station = None

    for stn_id, (lat, lon) in KMA_STATIONS.items():
        # Check if station is in dataframe
        station_data = df_weather[df_weather['STN'] == stn_id]
        if len(station_data) == 0:
            continue

        # Calculate distance
        x, y = latlon_to_raster_crs(lat, lon)
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        if distance < min_distance:
            min_distance = distance
            closest_station = station_data.iloc[0]

    if closest_station is None:
        print("No weather stations found, using zeros")
        return weather_channels

    # Extract weather values (use first valid station)
    temp = closest_station.get('TA', 0)
    humid = closest_station.get('HM', 0)
    wind_speed = closest_station.get('WS1', 0)
    wind_dir = closest_station.get('WD1', 0)
    precip = closest_station.get('RN-15m', 0)
    pressure = closest_station.get('PA', 0)
    dew_point = closest_station.get('TD', 0)

    # Handle NaN values
    temp = 0 if pd.isna(temp) else temp
    humid = 50 if pd.isna(humid) else humid
    wind_speed = 0 if pd.isna(wind_speed) else wind_speed
    wind_dir = 0 if pd.isna(wind_dir) else wind_dir
    precip = 0 if pd.isna(precip) else precip
    pressure = 1000 if pd.isna(pressure) else pressure
    dew_point = 0 if pd.isna(dew_point) else dew_point

    # Normalize using training parameters (min-max scaling)
    temp_norm = np.clip((temp - WEATHER_NORM['temperature']['min']) /
                        (WEATHER_NORM['temperature']['max'] - WEATHER_NORM['temperature']['min']),
                        0, 1)
    humid_norm = np.clip((humid - WEATHER_NORM['humidity']['min']) /
                         (WEATHER_NORM['humidity']['max'] - WEATHER_NORM['humidity']['min']),
                         0, 1)
    wind_speed_norm = np.clip((wind_speed - WEATHER_NORM['wind_speed']['min']) /
                              (WEATHER_NORM['wind_speed']['max'] - WEATHER_NORM['wind_speed']['min']),
                              0, 1)
    wind_dir_norm = np.clip((wind_dir - WEATHER_NORM['wind_direction']['min']) /
                            (WEATHER_NORM['wind_direction']['max'] - WEATHER_NORM['wind_direction']['min']),
                            0, 1)
    precip_norm = np.clip((precip - WEATHER_NORM['precipitation']['min']) /
                          (WEATHER_NORM['precipitation']['max'] - WEATHER_NORM['precipitation']['min']),
                          0, 1)
    pressure_norm = np.clip((pressure - WEATHER_NORM['pressure']['min']) /
                            (WEATHER_NORM['pressure']['max'] - WEATHER_NORM['pressure']['min']),
                            0, 1)
    dew_point_norm = np.clip((dew_point - WEATHER_NORM['dew_point']['min']) /
                             (WEATHER_NORM['dew_point']['max'] - WEATHER_NORM['dew_point']['min']),
                             0, 1)

    # Fill entire grid with closest station's data
    weather_channels[0, :, :] = temp_norm        # Ch 2: Temperature
    weather_channels[1, :, :] = humid_norm       # Ch 3: Humidity
    weather_channels[2, :, :] = wind_speed_norm  # Ch 4: Wind Speed
    weather_channels[3, :, :] = wind_dir_norm    # Ch 5: Wind Direction
    weather_channels[4, :, :] = precip_norm      # Ch 6: Precipitation
    weather_channels[5, :, :] = pressure_norm    # Ch 7: Pressure
    weather_channels[6, :, :] = 0.0              # Ch 8: Cloud Cover (always 0)
    weather_channels[7, :, :] = 0.0              # Ch 9: Visibility (always 0)
    weather_channels[8, :, :] = dew_point_norm   # Ch 10: Dew Point

    print(f"Using weather from closest station (distance: {min_distance/1000:.1f} km)")
    print(f"  Temperature: {temp:.1f}C, Humidity: {humid:.1f}%, Wind: {wind_speed:.1f}m/s")

    return weather_channels


def create_input_tensor(static_channels, weather_channels, fire_mask):
    """
    Create 16-channel environmental input + fire mask

    Args:
        static_channels: (7, 30, 30) - slope, aspect, ndvi, fsm(4)
        weather_channels: (9, 30, 30) - weather data
        fire_mask: (30, 30) - initial fire

    Returns:
        input_data: (17, 30, 30) numpy array ready for model
                   Channel order: [slope, aspect, weather(9), ndvi, fsm(4), fire_mask]
    """
    # Stack channels in correct order
    # static_channels = [slope(0), aspect(1), ndvi(2), fsm(3-6)]
    # weather_channels = [temp, humid, wind_speed, wind_dir, precip, pressure, cloud, vis, dew]

    input_data = np.concatenate([
        static_channels[0:2],    # Ch 0-1: Slope, Aspect
        weather_channels,         # Ch 2-10: Weather (9 channels)
        static_channels[2:3],    # Ch 11: NDVI
        static_channels[3:7],    # Ch 12-15: FSM (4 channels)
        fire_mask[np.newaxis, :, :]  # Ch 16: Fire mask
    ], axis=0)  # (17, 30, 30)

    return input_data
