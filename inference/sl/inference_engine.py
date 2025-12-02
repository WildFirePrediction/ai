"""
Inference engine for U-Net wildfire prediction
Handles model loading and prediction
"""
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from sl_training.unet_16ch_v3.model import UNetMultiTimestep


class WildfireInferenceEngine:
    """Manages model loading and inference"""

    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize inference engine

        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            device: Device for inference ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.model = self._load_model()

        print(f"Inference engine initialized")
        print(f"  Device: {self.device}")
        print(f"  Checkpoint: {self.checkpoint_path}")

    def _load_model(self):
        """Load trained U-Net V3 model"""
        print(f"Loading model from {self.checkpoint_path}...")

        # Create model instance (17 input channels, 3 output timesteps)
        model = UNetMultiTimestep(n_channels=17, n_timesteps=3)

        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])

        # Move to device and set to eval mode
        model.to(self.device)
        model.eval()

        # Print checkpoint info
        best_iou = checkpoint.get('best_iou', 'N/A')
        epoch = checkpoint.get('epoch', 'N/A')
        print(f"  Model loaded: Epoch {epoch}, Best IoU: {best_iou}")

        return model

    def predict(self, input_data):
        """
        Run inference on input data

        Args:
            input_data: (17, 30, 30) numpy array
                       16 environmental channels + 1 fire mask

        Returns:
            predictions: (3, 30, 30) numpy array
                        Predicted probabilities for t+1, t+2, t+3
        """
        # Convert to tensor and add batch dimension
        input_tensor = torch.from_numpy(input_data).float().unsqueeze(0).to(self.device)
        # Shape: (1, 17, 30, 30)

        # Run inference
        with torch.no_grad():
            logits = self.model(input_tensor)  # (1, 3, 30, 30)
            predictions = torch.sigmoid(logits).cpu().numpy()[0]  # (3, 30, 30)

        return predictions

    def process_predictions(self, predictions, grid_coords, fire_timestamp,
                          threshold=0.5, timestep_hours=1):
        """
        Convert predictions to lat/lon coordinates with timestamps

        Args:
            predictions: (3, 30, 30) predicted probabilities
            grid_coords: (30, 30, 2) grid coordinates in raster CRS
            fire_timestamp: datetime object of fire start
            threshold: Probability threshold for prediction (default 0.5)
            timestep_hours: Hours per timestep (default 1)

        Returns:
            results: List of dicts per timestep with predicted cells
        """
        from datetime import timedelta
        from inference.sl.grid_utils import raster_crs_to_latlon

        results = []

        for t in range(3):  # t+1, t+2, t+3
            # Find cells above threshold
            pred_cells = np.where(predictions[t] > threshold)

            predicted_cells = []
            for i in range(len(pred_cells[0])):
                row, col = pred_cells[0][i], pred_cells[1][i]
                prob = float(predictions[t, row, col])

                # Get cell coordinates
                x, y = grid_coords[row, col]
                lat, lon = raster_crs_to_latlon(x, y)

                predicted_cells.append({
                    'lat': float(lat),
                    'lon': float(lon),
                    'probability': prob
                })

            # Calculate timestamp for this timestep
            pred_timestamp = fire_timestamp + timedelta(hours=timestep_hours * (t + 1))

            results.append({
                'timestep': t + 1,
                'timestamp': pred_timestamp.isoformat(),
                'predicted_cells': predicted_cells
            })

        return results
