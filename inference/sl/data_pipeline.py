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

# KMA weather station coordinates (subset - full list can be added)
KMA_STATIONS = {
    90: (37.5714, 126.9658),    # Seoul
    92: (35.1041, 129.0320),    # Busan
    93: (37.4563, 126.7052),    # Incheon
    95: (35.8714, 128.6014),    # Daegu
    98: (35.1595, 126.8526),    # Gwangju
    99: (35.5383, 129.3114),    # Ulsan
    100: (36.3667, 127.3833),   # Daejeon
    108: (37.9838, 127.0288),   # Uijeongbu
    112: (37.2636, 127.0286),   # Suwon
    114: (37.7480, 128.8760),   # Gangneung
    115: (37.8813, 127.7298),   # Chuncheon
    119: (35.8279, 127.1480),   # Jeonju
    121: (36.7995, 127.0057),   # Cheongju
    127: (35.1064, 126.8906),   # Mokpo
    129: (36.0190, 129.3435),   # Pohang
    130: (34.8161, 126.3916),   # Yeosu
    133: (34.4736, 126.6228),   # Heuksando
    136: (33.5141, 126.5292),   # Jeju
    137: (33.2894, 126.5653),   # Seogwipo
    138: (33.3822, 126.2997),   # Gosan
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

        # Replace NaN values with defaults (handles missing data in rasters)
        slope = np.nan_to_num(slope, nan=0.0)
        aspect = np.nan_to_num(aspect, nan=0.0)
        ndvi = np.nan_to_num(ndvi, nan=0.0)  # NDVI often has NaN for water/urban
        fsm = np.nan_to_num(fsm, nan=1.0)  # Default to FSM class 1

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
