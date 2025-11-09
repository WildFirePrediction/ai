"""A3C implementation for wildfire prediction with parallel CPU workers."""

from .model import A3C_SpatialFireModel
from .worker import worker_process

__all__ = ['A3C_SpatialFireModel', 'worker_process']
