"""
Web demo inference wrapper around the production RL pipeline.

Loads the A3C checkpoint once and exposes a single predict() call that takes an
ignition point (lat/lon) and returns predictions plus the extra context the web
UI shows (weather, terrain, spread statistics).

Reuses the exact production modules:
    inference.rl.inference_engine  - A3C model + iterative rollout
    inference.rl.data_pipeline     - static raster + KMA weather -> 16 channels
    inference.rl.grid_utils        - 30x30 / 400m grid and CRS conversions
"""
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Project root and inference/rl must both be importable:
# inference_engine.py does "from rlconfig import TIMESTEPS" (module lives in inference/rl)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'inference' / 'rl'))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from config import KMA_AWS_BASE_URL                                # noqa: E402
from rlconfig import TIMESTEPS                                     # noqa: E402
from inference.rl.inference_engine import WildfireRLInferenceEngine  # noqa: E402
from inference.rl.data_pipeline import (                           # noqa: E402
    StaticDataLoader,
    fetch_kma_weather,
    process_weather_data,
    create_rl_input_tensor,
    KMA_STATIONS,
)
from inference.rl.grid_utils import (                              # noqa: E402
    create_fire_grid,
    create_initial_fire_mask,
    latlon_to_raster_crs,
    raster_crs_to_latlon,
    get_raster_crs,
)

# Grid geometry (must match training / production inference)
GRID_SIZE = 30
CELL_SIZE_M = 400
TIMESTEP_MINUTES = 10

# Fuel model classes of the FSM raster (channel 12-15 one-hot)
FSM_CLASS_NAMES = {
    0: 'Conifer',
    1: 'Broadleaf',
    2: 'Mixed',
    3: 'Non-forest',
}

# Weather cache lifetime: repeated ignition points in one run share a fetch
WEATHER_CACHE_TTL_S = 300


class WebDemoEngine:
    """Single-model inference engine driving the interactive web demo."""

    def __init__(self, checkpoint_path, data_dir='embedded_data',
                 output_dir='webdemo/outputs', device='cuda'):
        self.checkpoint_path = str(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Serialises GPU rollout + rasterio window reads (neither is thread safe)
        self.lock = threading.Lock()

        self.engine = WildfireRLInferenceEngine(
            checkpoint_path=checkpoint_path,
            device=device,
            sequence_length=3,
        )
        self.device = self.engine.device
        self.static_loader = StaticDataLoader(data_dir=data_dir)

        # Raster coverage in the native CRS, used to reject out-of-coverage clicks
        self.raster_bounds = self.static_loader.dem_raster.bounds

        self._weather_cache = {}   # {"YYYYMMDDHHMM": (fetch_time, dataframe)}

        print(f"Web demo engine ready (device={self.device})")

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------
    def check_coverage(self, lat, lon):
        """
        Verify the full 12km x 12km inference grid fits inside the raster extent.

        Returns:
            (ok: bool, message: str)
        """
        try:
            x, y = latlon_to_raster_crs(lat, lon)
        except Exception as exc:
            return False, f"Coordinate conversion failed: {exc}"

        half = (GRID_SIZE * CELL_SIZE_M) / 2
        b = self.raster_bounds
        inside = (b.left <= x - half and x + half <= b.right and
                  b.bottom <= y - half and y + half <= b.top)

        if not inside:
            return False, "Point is outside the environmental data coverage"
        return True, "ok"

    # ------------------------------------------------------------------
    # Weather
    # ------------------------------------------------------------------
    def _get_weather_frame(self, weather_timestamp):
        """Fetch KMA observations for a timestamp, cached for WEATHER_CACHE_TTL_S."""
        key = weather_timestamp.strftime("%Y%m%d%H%M")
        now = time.time()

        cached = self._weather_cache.get(key)
        if cached and (now - cached[0]) < WEATHER_CACHE_TTL_S:
            return cached[1]

        df = fetch_kma_weather(weather_timestamp, KMA_AWS_BASE_URL)
        self._weather_cache[key] = (now, df)
        return df

    @staticmethod
    def _nearest_station_summary(df_weather, center_xy, weather_timestamp):
        """
        Human readable weather readout for the station actually used by the model.

        Mirrors the station selection in process_weather_data() so the UI shows
        exactly the observation that fed the 16-channel tensor.
        """
        summary = {
            'available': False,
            'timestamp': weather_timestamp.isoformat(),
        }
        if df_weather is None or len(df_weather) == 0:
            return summary

        center_x, center_y = center_xy
        best_distance = float('inf')
        best_row = None
        best_stn = None

        for stn_id, (lat, lon) in KMA_STATIONS.items():
            rows = df_weather[df_weather['STN'] == stn_id]
            if len(rows) == 0:
                continue
            x, y = latlon_to_raster_crs(lat, lon)
            distance = float(np.hypot(x - center_x, y - center_y))
            if distance < best_distance:
                best_distance = distance
                best_row = rows.iloc[0]
                best_stn = stn_id

        if best_row is None:
            return summary

        def value(column, default=None):
            raw = best_row.get(column, default)
            if raw is None or pd.isna(raw):
                return None
            return float(raw)

        summary.update({
            'available': True,
            'station_id': int(best_stn),
            'station_lat': KMA_STATIONS[best_stn][0],
            'station_lon': KMA_STATIONS[best_stn][1],
            'station_distance_km': round(best_distance / 1000.0, 1),
            'temperature_c': value('TA'),
            'humidity_pct': value('HM'),
            'wind_speed_ms': value('WS1'),
            'wind_direction_deg': value('WD1'),
            'precipitation_mm': value('RN-15m'),
            'pressure_hpa': value('PA'),
            'dew_point_c': value('TD'),
        })
        return summary

    # ------------------------------------------------------------------
    # Terrain
    # ------------------------------------------------------------------
    @staticmethod
    def _terrain_summary(static_channels):
        """Slope / aspect / NDVI / dominant fuel model over the inference grid."""
        center = GRID_SIZE // 2
        fsm_onehot = static_channels[3:7]                       # (4, 30, 30)
        fsm_share = fsm_onehot.reshape(4, -1).mean(axis=1)
        dominant = int(np.argmax(fsm_share))

        return {
            'slope_deg_center': round(float(static_channels[0, center, center]), 1),
            'slope_deg_mean': round(float(np.mean(static_channels[0])), 1),
            'aspect_deg_center': round(float(static_channels[1, center, center]), 1),
            'ndvi_center': round(float(static_channels[2, center, center]), 3),
            'ndvi_mean': round(float(np.mean(static_channels[2])), 3),
            'fuel_model': FSM_CLASS_NAMES.get(dominant, 'Unknown'),
            'fuel_model_share': round(float(fsm_share[dominant]), 2),
        }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    @staticmethod
    def _spread_statistics(results, fire_lat, fire_lon):
        """
        Per-timestep and overall spread metrics derived from predicted cells.

        Area uses the 400m cell footprint (0.16 km^2 per cell).
        """
        cell_area_km2 = (CELL_SIZE_M / 1000.0) ** 2
        origin_x, origin_y = latlon_to_raster_crs(fire_lat, fire_lon)

        per_timestep = []
        cumulative_cells = 1                        # ignition cell itself
        max_distance_m = 0.0

        for step in results:
            new_cells = step['predicted_cells']
            cumulative_cells += len(new_cells)

            for cell in new_cells:
                x, y = latlon_to_raster_crs(cell['lat'], cell['lon'])
                max_distance_m = max(max_distance_m, float(np.hypot(x - origin_x, y - origin_y)))

            per_timestep.append({
                'timestep': step['timestep'],
                'timestamp': step['timestamp'],
                'minutes': step['timestep'] * TIMESTEP_MINUTES,
                'new_cells': len(new_cells),
                'cumulative_cells': cumulative_cells,
                'area_km2': round(cumulative_cells * cell_area_km2, 3),
            })

        total_minutes = len(results) * TIMESTEP_MINUTES
        return {
            'per_timestep': per_timestep,
            'total_cells': cumulative_cells,
            'total_area_km2': round(cumulative_cells * cell_area_km2, 3),
            'total_area_ha': round(cumulative_cells * cell_area_km2 * 100, 1),
            'max_spread_m': round(max_distance_m, 1),
            'mean_rate_of_spread_m_per_min': (
                round(max_distance_m / total_minutes, 2) if total_minutes else 0.0
            ),
            'horizon_minutes': total_minutes,
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, fire_id, lat, lon, fire_timestamp=None, num_timesteps=TIMESTEPS,
                save=True):
        """
        Run the full inference pipeline for one ignition point.

        Args:
            fire_id: Identifier used for the saved output file
            lat, lon: Ignition point (WGS84)
            fire_timestamp: datetime of ignition (defaults to now)
            num_timesteps: Number of 10-minute steps to roll out
            save: Persist the result JSON under output_dir

        Returns:
            dict shaped like the production /predict payload plus web demo extras
        """
        started = time.time()
        fire_lat = float(lat)
        fire_lon = float(lon)
        fire_timestamp = fire_timestamp or datetime.now()
        num_timesteps = int(num_timesteps)

        # Weather API lags observations by 1-2 minutes (same offset as production)
        weather_timestamp = fire_timestamp - timedelta(minutes=3)

        with self.lock:
            grid_bounds, grid_coords, center_xy = create_fire_grid(
                fire_lat, fire_lon, grid_size=GRID_SIZE, cell_size=CELL_SIZE_M
            )
            static_channels = self.static_loader.extract_static_features(grid_bounds)

            df_weather = self._get_weather_frame(weather_timestamp)
            weather_channels = process_weather_data(df_weather, center_xy, grid_size=GRID_SIZE)
            env_data = create_rl_input_tensor(static_channels, weather_channels)
            initial_fire_mask = create_initial_fire_mask(center_xy, grid_coords,
                                                         grid_size=GRID_SIZE)

            predictions = self.engine.predict_iterative(
                env_data=env_data,
                initial_fire_mask=initial_fire_mask,
                num_timesteps=num_timesteps,
            )
            results = self.engine.process_predictions(
                predictions=predictions,
                initial_fire_mask=initial_fire_mask,
                grid_coords=grid_coords,
                fire_timestamp=fire_timestamp,
                timestep_hours=TIMESTEP_MINUTES / 60.0,
            )

            weather = self._nearest_station_summary(df_weather, center_xy, weather_timestamp)
            terrain = self._terrain_summary(static_channels)

        # Grid extent as lat/lon corners so the UI can outline the model domain
        x_min, x_max, y_min, y_max = grid_bounds
        south, west = raster_crs_to_latlon(x_min, y_min)
        north, east = raster_crs_to_latlon(x_max, y_max)

        output = {
            'event_type': '0',
            'fire_id': str(fire_id),
            'fire_location': {'lat': fire_lat, 'lon': fire_lon},
            'fire_timestamp': fire_timestamp.isoformat(),
            'inference_timestamp': datetime.now().isoformat(),
            'model': 'a3c_16ch_v3_lstm_rel',
            'predictions': results,
            'grid': {
                'size': GRID_SIZE,
                'cell_size_m': CELL_SIZE_M,
                'timestep_minutes': TIMESTEP_MINUTES,
                'bounds': {'south': south, 'west': west, 'north': north, 'east': east},
            },
            'weather': weather,
            'terrain': terrain,
            'statistics': self._spread_statistics(results, fire_lat, fire_lon),
            'runtime_ms': int((time.time() - started) * 1000),
        }

        if save:
            filename = f"fire_{fire_id}_{fire_timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            path = self.output_dir / filename
            with open(path, 'w') as handle:
                json.dump(output, handle, indent=2)
            output['saved_to'] = str(path)

        return output

    # ------------------------------------------------------------------
    def info(self):
        """Static engine metadata for the UI header."""
        return {
            'model': 'a3c_16ch_v3_lstm_rel',
            'checkpoint': self.checkpoint_path,
            'device': self.device,
            'crs': str(get_raster_crs()),
            'grid_size': GRID_SIZE,
            'cell_size_m': CELL_SIZE_M,
            'timestep_minutes': TIMESTEP_MINUTES,
            'default_timesteps': TIMESTEPS,
        }
