"""
Grid utilities for RL wildfire inference
Reuses coordinate conversion logic from SL inference
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Reuse all functions from SL grid_utils
from inference.sl.grid_utils import (
    get_raster_crs,
    get_transformers,
    latlon_to_raster_crs,
    raster_crs_to_latlon,
    create_fire_grid,
    grid_cell_to_latlon,
    create_initial_fire_mask
)

__all__ = [
    'get_raster_crs',
    'get_transformers',
    'latlon_to_raster_crs',
    'raster_crs_to_latlon',
    'create_fire_grid',
    'grid_cell_to_latlon',
    'create_initial_fire_mask'
]
