"""
Data pipeline for RL wildfire inference
Reuses static data extraction and weather processing from SL inference
Key difference: RL uses 16 channels (fire mask is separate input)
"""
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Reuse data loading functions
from inference.sl.data_pipeline import (
    StaticDataLoader,
    fetch_kma_weather,
    process_weather_data,
    WEATHER_NORM,
    KMA_STATIONS
)

__all__ = [
    'StaticDataLoader',
    'fetch_kma_weather',
    'process_weather_data',
    'create_rl_input_tensor',
    'WEATHER_NORM',
    'KMA_STATIONS'
]


def create_rl_input_tensor(static_channels, weather_channels):
    """
    Create 16-channel environmental input for RL model (fire mask separate)

    Args:
        static_channels: (7, 30, 30) - slope, aspect, ndvi, fsm(4)
        weather_channels: (9, 30, 30) - weather data

    Returns:
        input_data: (16, 30, 30) numpy array ready for RL model
                   Channel order: [slope, aspect, weather(9), ndvi, fsm(4)]
                   Note: Fire mask is NOT included (separate input to model)
    """
    # Stack channels in correct order
    # static_channels = [slope(0), aspect(1), ndvi(2), fsm(3-6)]
    # weather_channels = [temp, humid, wind_speed, wind_dir, precip, pressure, cloud, vis, dew]

    input_data = np.concatenate([
        static_channels[0:2],    # Ch 0-1: Slope, Aspect
        weather_channels,         # Ch 2-10: Weather (9 channels)
        static_channels[2:3],    # Ch 11: NDVI
        static_channels[3:7],    # Ch 12-15: FSM (4 channels)
    ], axis=0)  # (16, 30, 30)

    return input_data
