"""
Inference engine for A3C wildfire prediction
Handles model loading and iterative prediction
"""
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from rl_training.a3c_16ch.V3_LSTM_REL.model import A3C_PerCellModel_LSTM


class WildfireRLInferenceEngine:
    """Manages A3C model loading and iterative inference"""

    def __init__(self, checkpoint_path, device='cuda', sequence_length=3):
        """
        Initialize RL inference engine

        Args:
            checkpoint_path: Path to A3C model checkpoint (.pt file)
            device: Device for inference ('cuda' or 'cpu')
            sequence_length: LSTM sequence length (default 3)
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.checkpoint_path = Path(checkpoint_path)
        self.sequence_length = sequence_length

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.model = self._load_model()

        print(f"RL Inference engine initialized")
        print(f"  Device: {self.device}")
        print(f"  Checkpoint: {self.checkpoint_path}")

    def _load_model(self):
        """Load trained A3C model"""
        print(f"Loading A3C model from {self.checkpoint_path}...")

        # Create model instance (16 channels, LSTM hidden=256, sequence_length=3)
        model = A3C_PerCellModel_LSTM(
            in_channels=16,
            lstm_hidden=256,
            sequence_length=self.sequence_length,
            use_groupnorm=True
        )

        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Some checkpoints save model directly
            model.load_state_dict(checkpoint)

        # Move to device and set to eval mode
        model.to(self.device)
        model.eval()

        # Print checkpoint info
        best_iou = checkpoint.get('best_iou', 'N/A') if isinstance(checkpoint, dict) else 'N/A'
        episode = checkpoint.get('episode', 'N/A') if isinstance(checkpoint, dict) else 'N/A'
        print(f"  Model loaded: Episode {episode}, Best IoU: {best_iou}")

        return model

    def predict_iterative(self, env_data, initial_fire_mask, num_timesteps=5):
        """
        Run iterative inference for multiple timesteps

        Args:
            env_data: (16, 30, 30) numpy array
                     16 environmental channels (NO fire mask)
            initial_fire_mask: (30, 30) numpy array
                              Binary mask with 1.0 at fire origin
            num_timesteps: Number of future timesteps to predict (default 5)

        Returns:
            predictions: List of (30, 30) binary arrays for t+1, t+2, ..., t+N
                        Each array shows cumulative fire spread
        """
        # Validate input data for NaN/inf values
        if np.any(np.isnan(env_data)):
            nan_count = np.sum(np.isnan(env_data))
            nan_channels = np.where(np.any(np.isnan(env_data.reshape(16, -1)), axis=1))[0]
            raise ValueError(f"env_data contains {nan_count} NaN values in channels: {nan_channels}")

        if np.any(np.isinf(env_data)):
            inf_count = np.sum(np.isinf(env_data))
            inf_channels = np.where(np.any(np.isinf(env_data.reshape(16, -1)), axis=1))[0]
            raise ValueError(f"env_data contains {inf_count} inf values in channels: {inf_channels}")

        if np.any(np.isnan(initial_fire_mask)):
            raise ValueError(f"initial_fire_mask contains NaN values")

        # Print data ranges for debugging
        print(f"  Data validation passed:")
        print(f"    env_data shape: {env_data.shape}, range: [{env_data.min():.3f}, {env_data.max():.3f}]")
        print(f"    fire_mask shape: {initial_fire_mask.shape}, range: [{initial_fire_mask.min():.3f}, {initial_fire_mask.max():.3f}]")

        # Convert to tensors
        env_tensor = torch.from_numpy(env_data).float().to(self.device)
        # Shape: (16, 30, 30)

        # Create temporal sequence by repeating environmental data
        # (sequence_length, 16, 30, 30)
        sequence = env_tensor.unsqueeze(0).repeat(self.sequence_length, 1, 1, 1)
        # Add batch dimension: (1, sequence_length, 16, 30, 30)
        sequence = sequence.unsqueeze(0)

        # Initialize fire mask
        current_fire_mask = torch.from_numpy(initial_fire_mask).float().to(self.device)
        # Shape: (30, 30)

        predictions = []

        # Iteratively predict for each timestep
        for t in range(num_timesteps):
            # Add batch dimension to fire mask: (1, 30, 30)
            fire_mask_batch = current_fire_mask.unsqueeze(0)

            # Run model inference
            with torch.no_grad():
                try:
                    action_grid, log_prob, entropy, value, burning_cells_info = \
                        self.model.get_action_and_value(sequence, fire_mask_batch, action=None)
                except RuntimeError as e:
                    # Handle CUDA assertion errors (probabilities out of range)
                    print(f"  WARNING: Model prediction failed at timestep t+{t+1}: {e}")
                    print(f"  This may be due to numerical instability or out-of-distribution inputs.")
                    # Return empty prediction for this and remaining timesteps
                    break

            # action_grid: (30, 30) - binary prediction of NEW burning cells
            # Convert to numpy
            new_fire_cells = action_grid.cpu().numpy()

            # Update fire mask: combine current fire + new fire
            # This simulates fire persistence
            current_fire_mask = torch.maximum(
                current_fire_mask,
                action_grid.to(self.device)
            )

            # Store cumulative fire state for this timestep
            predictions.append(current_fire_mask.cpu().numpy())

            print(f"  Timestep t+{t+1}: {int(new_fire_cells.sum())} new cells, "
                  f"{int(current_fire_mask.sum().item())} total burning cells")

        return predictions

    def process_predictions(self, predictions, initial_fire_mask, grid_coords,
                          fire_timestamp, timestep_hours=1):
        """
        Convert binary predictions to lat/lon coordinates with timestamps

        Args:
            predictions: List of (30, 30) binary arrays
            initial_fire_mask: (30, 30) initial fire state
            grid_coords: (30, 30, 2) grid coordinates in raster CRS
            fire_timestamp: datetime object of fire start
            timestep_hours: Hours per timestep (default 1)

        Returns:
            results: List of dicts per timestep with predicted cells
        """
        from datetime import timedelta
        from inference.rl.grid_utils import raster_crs_to_latlon

        results = []

        for t, pred_mask in enumerate(predictions):
            # Find NEW cells at this timestep (not in initial or previous)
            if t == 0:
                # t+1: new cells = pred_mask - initial_fire_mask
                previous_mask = initial_fire_mask
            else:
                # t+2, t+3: new cells = current_mask - previous_mask
                previous_mask = predictions[t-1]

            # NEW cells only (exclude already burning cells)
            new_cells_mask = (pred_mask > 0.5) & (previous_mask < 0.5)
            new_cell_indices = np.where(new_cells_mask)

            predicted_cells = []
            for i in range(len(new_cell_indices[0])):
                row, col = new_cell_indices[0][i], new_cell_indices[1][i]

                # Get cell coordinates
                x, y = grid_coords[row, col]
                lat, lon = raster_crs_to_latlon(x, y)

                predicted_cells.append({
                    'lat': float(lat),
                    'lon': float(lon),
                    'probability': 1.0  # RL gives binary predictions
                })

            # Calculate timestamp for this timestep
            pred_timestamp = fire_timestamp + timedelta(hours=timestep_hours * (t + 1))

            results.append({
                'timestep': t + 1,
                'timestamp': pred_timestamp.isoformat(),
                'predicted_cells': predicted_cells
            })

        return results
