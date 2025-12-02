"""
Grid utilities for wildfire inference
Handles coordinate conversion and grid creation
Uses raster's native CRS (custom Transverse Mercator)
"""
import numpy as np
import pyproj
import rasterio
from pathlib import Path


# Cached transformers and raster CRS
_transformer_to_raster = None
_transformer_to_wgs84 = None
_raster_crs = None

def get_raster_crs():
    """Get raster CRS from DEM file (cached)"""
    global _raster_crs

    if _raster_crs is None:
        dem_path = Path('embedded_data/dem_slope_aspect_FINAL.tif')
        with rasterio.open(dem_path) as src:
            _raster_crs = src.crs

    return _raster_crs


def get_transformers():
    """Get or create coordinate transformers (cached)"""
    global _transformer_to_raster, _transformer_to_wgs84

    raster_crs = get_raster_crs()

    if _transformer_to_raster is None:
        _transformer_to_raster = pyproj.Transformer.from_crs(
            "EPSG:4326", raster_crs, always_xy=True
        )

    if _transformer_to_wgs84 is None:
        _transformer_to_wgs84 = pyproj.Transformer.from_crs(
            raster_crs, "EPSG:4326", always_xy=True
        )

    return _transformer_to_raster, _transformer_to_wgs84


def latlon_to_raster_crs(lat, lon):
    """
    Convert WGS84 (lat, lon) to raster CRS (x, y)

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees

    Returns:
        x, y: Coordinates in raster CRS (meters)
    """
    transformer, _ = get_transformers()
    x, y = transformer.transform(lon, lat)  # Note: always_xy=True means (lon, lat)
    return x, y


def raster_crs_to_latlon(x, y):
    """
    Convert raster CRS (x, y) to WGS84 (lat, lon)

    Args:
        x, y: Coordinates in raster CRS (meters)

    Returns:
        lat, lon: Latitude and longitude in decimal degrees
    """
    _, transformer = get_transformers()
    lon, lat = transformer.transform(x, y)  # Returns (lon, lat)
    return lat, lon


def create_fire_grid(center_lat, center_lon, grid_size=30, cell_size=400):
    """
    Create 30x30 grid centered at fire location

    Args:
        center_lat: Fire latitude (WGS84)
        center_lon: Fire longitude (WGS84)
        grid_size: Grid dimensions (default 30x30)
        cell_size: Cell resolution in meters (default 400m)

    Returns:
        grid_bounds: (x_min, x_max, y_min, y_max) in raster CRS
        grid_coords: (30, 30, 2) array of (x, y) in raster CRS for each cell center
        center_xy: (center_x, center_y) in raster CRS
    """
    # Convert center to raster CRS
    center_x, center_y = latlon_to_raster_crs(center_lat, center_lon)

    # Calculate grid extent (6km radius = 12km x 12km)
    half_extent = (grid_size * cell_size) / 2  # 6000m = 6km

    x_min = center_x - half_extent
    x_max = center_x + half_extent
    y_min = center_y - half_extent
    y_max = center_y + half_extent

    # Create grid coordinates (cell centers)
    # Note: Y coordinates go from y_max to y_min (raster convention: top to bottom)
    x_coords = np.linspace(x_min, x_max, grid_size)
    y_coords = np.linspace(y_max, y_min, grid_size)  # Reverse for raster

    X, Y = np.meshgrid(x_coords, y_coords)

    grid_coords = np.stack([X, Y], axis=-1)  # (30, 30, 2)
    grid_bounds = (x_min, x_max, y_min, y_max)
    center_xy = (center_x, center_y)

    return grid_bounds, grid_coords, center_xy


def grid_cell_to_latlon(row, col, grid_coords):
    """
    Convert grid cell (row, col) to WGS84 (lat, lon)

    Args:
        row: Grid row index (0-29)
        col: Grid column index (0-29)
        grid_coords: (30, 30, 2) array from create_fire_grid()

    Returns:
        lat, lon: Cell center coordinates in WGS84
    """
    x, y = grid_coords[row, col]
    lat, lon = raster_crs_to_latlon(x, y)
    return lat, lon


def create_initial_fire_mask(center_xy, grid_coords, grid_size=30):
    """
    Create initial fire mask with single cell at fire center

    Args:
        center_xy: (center_x, center_y) fire location in EPSG:5179
        grid_coords: (30, 30, 2) array of grid coordinates
        grid_size: Grid dimensions (default 30)

    Returns:
        fire_mask: (30, 30) binary mask with 1.0 at fire center
    """
    fire_mask = np.zeros((grid_size, grid_size), dtype=np.float32)

    center_x, center_y = center_xy

    # Find closest grid cell to fire center
    distances = np.sqrt(
        (grid_coords[:, :, 0] - center_x)**2 +
        (grid_coords[:, :, 1] - center_y)**2
    )

    # Set closest cell to 1.0
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    fire_mask[min_idx] = 1.0

    return fire_mask
