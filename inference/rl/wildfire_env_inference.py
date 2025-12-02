"""
Wildfire Environment for Real-time Inference
Modified from training environment to support:
- 15 channels (matching v3r2 model: includes fire_age)
- Dynamic grid creation around fire location
- Real-time data integration
"""
import numpy as np
from typing import Dict, Tuple, Optional
import torch


class WildfireEnvInference:
    """
    Environment for real-time wildfire prediction inference.
    
    Key differences from training env:
    - 15 channels (includes fire_age) to match v3r2 model
    - Dynamic grid creation around fire trigger
    - Supports grid expansion when fire spreads beyond initial boundary
    """
    
    def __init__(
        self,
        static_features: np.ndarray,  # (3, H, W) - DEM, slope, aspect
        lcm: np.ndarray,              # (H, W) - land cover
        fsm: np.ndarray,              # (H, W) - fuel moisture
        initial_fire_mask: np.ndarray,  # (H, W) - initial fire location
        weather_data: np.ndarray,     # (6,) - weather vector
        resolution_m: int = 400,
        grid_origin: Optional[Tuple[int, int]] = None  # (row, col) in global grid
    ):
        """
        Initialize inference environment.
        
        Args:
            static_features: Static terrain features (DEM, slope, aspect)
            lcm: Land cover map
            fsm: Fuel moisture map
            initial_fire_mask: Initial fire locations (binary mask)
            weather_data: Weather vector [temp, humidity, wind_speed, wind_x, wind_y, rainfall]
            resolution_m: Grid resolution in meters
            grid_origin: Origin of this grid in global coordinate system
        """
        self.H, self.W = static_features.shape[1], static_features.shape[2]
        self.resolution_m = resolution_m
        self.grid_origin = grid_origin or (0, 0)
        
        # Store static data (handle NaN values)
        self.static_cont = static_features.astype(np.float32)  # (3, H, W)
        # Replace NaN with 0 (common in slope/aspect for flat areas or invalid data)
        self.static_cont = np.nan_to_num(self.static_cont, nan=0.0, posinf=0.0, neginf=0.0)
        self.lcm = lcm.astype(np.float32)
        self.fsm = fsm.astype(np.float32)
        
        # Normalize categorical features
        lcm_max = max(1, int(self.lcm.max()))
        fsm_max = max(1, int(self.fsm.max()))
        self.lcm_norm = self.lcm / float(lcm_max)
        self.fsm_norm = self.fsm / float(fsm_max)
        
        # Initialize fire state
        self.fire_mask = initial_fire_mask.astype(np.float32)  # (H, W)
        self.fire_intensity = np.zeros_like(self.fire_mask)    # (H, W)
        self.fire_temp = np.zeros_like(self.fire_mask)         # (H, W)
        self.fire_age = np.zeros_like(self.fire_mask)          # (H, W)
        
        # Set initial fire properties for burning cells
        burning = self.fire_mask > 0
        self.fire_intensity[burning] = 3.0  # Default intensity
        self.fire_temp[burning] = 20.0      # Default temp (Celsius)
        self.fire_age[burning] = 1.0        # Initial age (1 timestep)
        
        # Weather state
        self.weather = weather_data.astype(np.float32)  # (6,)
        
        # Timestep counter
        self.t = 0
        
        # Prediction history (for visualization and analysis)
        self.fire_history = [self.fire_mask.copy()]
        self.prediction_history = []
        
    def update_weather(self, weather_data: np.ndarray):
        """Update weather conditions (real-time update)."""
        self.weather = weather_data.astype(np.float32)
    
    def get_observation(self) -> np.ndarray:
        """
        Build current observation for model inference.
        
        Returns:
            obs: (15, H, W) observation tensor matching v3r2 model
        """
        # Static continuous (3)
        static_cont = self.static_cont  # (3, H, W)
        
        # Categorical (2)
        lcm = self.lcm_norm[None, ...]  # (1, H, W)
        fsm = self.fsm_norm[None, ...]  # (1, H, W)
        
        # Fire state (4) - includes fire_age for 15 channels
        fire_mask = self.fire_mask[None, ...]  # (1, H, W)
        fire_intensity = np.clip(self.fire_intensity / 5.0, 0, 1)[None, ...]  # (1, H, W)
        fire_temp = np.clip((self.fire_temp + 20) / 60.0, 0, 1)[None, ...]    # (1, H, W)
        fire_age = np.clip(self.fire_age / 1000.0, 0, 1)[None, ...]           # (1, H, W)
        
        # Weather (6) - broadcast to spatial dimensions
        w_normalized = np.zeros_like(self.weather)
        w_normalized[0] = np.clip((self.weather[0] + 10) / 50.0, 0, 1)  # temp
        w_normalized[1] = np.clip(self.weather[1] / 100.0, 0, 1)         # humidity
        w_normalized[2] = np.clip(self.weather[2] / 20.0, 0, 1)          # wind_speed
        w_normalized[3] = np.clip(self.weather[3] / 20.0, 0, 1)          # wind_x
        w_normalized[4] = np.clip(self.weather[4] / 20.0, 0, 1)          # wind_y
        w_normalized[5] = np.clip(self.weather[5] / 50.0, 0, 1)          # rainfall
        weather = np.tile(w_normalized[:, None, None], (1, self.H, self.W))  # (6, H, W)
        
        # Concatenate: 3 + 1 + 1 + 4 + 6 = 15 channels
        obs = np.concatenate([
            static_cont,      # (3, H, W)
            lcm,              # (1, H, W)
            fsm,              # (1, H, W)
            fire_mask,        # (1, H, W)
            fire_intensity,   # (1, H, W)
            fire_temp,        # (1, H, W)
            fire_age,         # (1, H, W)
            weather           # (6, H, W)
        ], axis=0).astype(np.float32)
        
        return obs
    
    def step(self, predicted_spread: np.ndarray):
        """
        Advance environment by one timestep based on predicted fire spread.
        
        Args:
            predicted_spread: (H, W) binary mask of predicted new burns
        """
        # Update fire mask (accumulate burns)
        new_burns = (predicted_spread > 0.5) & (self.fire_mask == 0)
        self.fire_mask = np.clip(self.fire_mask + new_burns.astype(np.float32), 0, 1)
        
        # Update fire properties for new burns
        self.fire_intensity[new_burns] = 3.0
        self.fire_temp[new_burns] = 20.0
        self.fire_age[new_burns] = 1.0
        
        # Increment age for existing fires
        existing_fires = (self.fire_mask > 0) & ~new_burns
        self.fire_age[existing_fires] += 1.0
        
        # Store in history
        self.fire_history.append(self.fire_mask.copy())
        self.prediction_history.append(predicted_spread.copy())
        
        # Increment timestep
        self.t += 1
    
    def reset_from_trigger(self, new_fire_mask: np.ndarray):
        """
        Hard reset environment with new fire trigger (for real-time updates).
        
        Args:
            new_fire_mask: New fire locations from updated fire detection
        """
        self.fire_mask = new_fire_mask.astype(np.float32)
        self.fire_intensity = np.zeros_like(self.fire_mask)
        self.fire_temp = np.zeros_like(self.fire_mask)
        self.fire_age = np.zeros_like(self.fire_mask)
        
        burning = self.fire_mask > 0
        self.fire_intensity[burning] = 3.0
        self.fire_temp[burning] = 20.0
        self.fire_age[burning] = 1.0
        
        self.t = 0
        self.fire_history = [self.fire_mask.copy()]
        self.prediction_history = []
    
    def check_boundary_overflow(self, margin: int = 10) -> Dict[str, bool]:
        """
        Check if fire has spread close to grid boundaries.
        Returns which boundaries need expansion.
        
        Args:
            margin: Distance from boundary to trigger expansion
            
        Returns:
            dict: {'north': bool, 'south': bool, 'east': bool, 'west': bool}
        """
        fire_locs = np.where(self.fire_mask > 0)
        
        if len(fire_locs[0]) == 0:
            return {'north': False, 'south': False, 'east': False, 'west': False}
        
        min_row, max_row = fire_locs[0].min(), fire_locs[0].max()
        min_col, max_col = fire_locs[1].min(), fire_locs[1].max()
        
        return {
            'north': min_row < margin,
            'south': max_row >= self.H - margin,
            'west': min_col < margin,
            'east': max_col >= self.W - margin
        }
    
    def get_fire_extent(self) -> Dict[str, int]:
        """Get current fire extent in grid coordinates."""
        fire_locs = np.where(self.fire_mask > 0)
        
        if len(fire_locs[0]) == 0:
            return {'min_row': -1, 'max_row': -1, 'min_col': -1, 'max_col': -1}
        
        return {
            'min_row': int(fire_locs[0].min()),
            'max_row': int(fire_locs[0].max()),
            'min_col': int(fire_locs[1].min()),
            'max_col': int(fire_locs[1].max())
        }
