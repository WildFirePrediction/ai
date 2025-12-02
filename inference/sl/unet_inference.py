"""
Simple inference script for U-Net V2 wildfire prediction
Visualizes predictions on validation episodes using Folium
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import folium
from folium import plugins
import random

# Import U-Net V2 model
from sl_training.unet_16ch_v2.model import UNetMultiTimestep


def load_model(checkpoint_path, device='cuda'):
    """Load trained U-Net V2 model"""
    print(f"Loading model from {checkpoint_path}")

    model = UNetMultiTimestep(n_channels=17, n_timesteps=3)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    best_iou = checkpoint.get('best_iou', 'N/A')
    epoch = checkpoint.get('epoch', 'N/A')
    print(f"Model loaded: Epoch {epoch}, Best IoU: {best_iou}")

    return model


def load_episode(episode_path):
    """Load a single episode from .npz file"""
    data = np.load(episode_path)
    states = data['states']  # (T, 16, 30, 30)
    fire_masks = data['fire_masks']  # (T, 30, 30)

    return states, fire_masks


def predict_episode(model, states, fire_masks, start_t, device='cuda'):
    """
    Run inference on episode starting from timestep start_t

    Args:
        model: Trained U-Net V2 model
        states: (T, 16, 30, 30) environment states
        fire_masks: (T, 30, 30) fire masks
        start_t: Starting timestep for prediction
        device: Device for inference

    Returns:
        predictions: (3, 30, 30) predicted new burns at t+1, t+2, t+3
        ground_truth: (3, 30, 30) actual new burns
    """
    # Create input: 16 env channels + 1 fire mask at time t
    state_t = states[start_t]  # (16, 30, 30)
    fire_t = fire_masks[start_t]  # (30, 30)

    input_data = np.concatenate([
        state_t,
        fire_t[np.newaxis, :, :]
    ], axis=0)  # (17, 30, 30)

    # Convert to tensor
    input_tensor = torch.from_numpy(input_data).float().unsqueeze(0).to(device)  # (1, 17, 30, 30)

    # Run inference
    with torch.no_grad():
        logits = model(input_tensor)  # (1, 3, 30, 30)
        predictions = torch.sigmoid(logits).cpu().numpy()[0]  # (3, 30, 30)

    # Get ground truth new burns
    ground_truth = []
    for dt in range(1, 4):
        if start_t + dt >= len(fire_masks):
            # Not enough future timesteps
            ground_truth.append(np.zeros((30, 30), dtype=np.float32))
        else:
            fire_prev = fire_masks[start_t + dt - 1]
            fire_next = fire_masks[start_t + dt]

            actual_mask_prev = fire_prev > 0.5
            actual_mask_next = fire_next > 0.5
            new_burns = (actual_mask_next & ~actual_mask_prev).astype(np.float32)
            ground_truth.append(new_burns)

    ground_truth = np.stack(ground_truth, axis=0)  # (3, 30, 30)

    return predictions, ground_truth, fire_t


def compute_iou(pred, target, threshold=0.5):
    """Compute IoU for each timestep"""
    ious = []
    for t in range(pred.shape[0]):
        pred_binary = (pred[t] > threshold).astype(np.float32)
        target_binary = target[t]

        intersection = (pred_binary * target_binary).sum()
        union = ((pred_binary + target_binary) > 0).astype(np.float32).sum()

        iou = (intersection + 1e-8) / (union + 1e-8)
        ious.append(iou)

    return ious


def compute_relaxed_iou(pred, target, threshold=0.5):
    """
    Compute relaxed IoU with 8-neighbor tolerance (dilated target)
    Matches V3 training metric
    """
    from scipy.ndimage import binary_dilation

    ious = []
    structure = np.ones((3, 3), dtype=bool)  # 3x3 dilation kernel

    for t in range(pred.shape[0]):
        pred_binary = (pred[t] > threshold).astype(bool)
        target_binary = (target[t] > 0.5).astype(bool)

        # Dilate target for 8-neighbor tolerance
        target_dilated = binary_dilation(target_binary, structure=structure)

        intersection = (pred_binary & target_dilated).sum()
        union = (pred_binary | target_dilated).sum()

        iou = (intersection + 1e-8) / (union + 1e-8)
        ious.append(iou)

    return ious


def create_visualization(predictions, ground_truth, initial_fire, episode_name, output_path):
    """
    Create Folium map showing predictions vs ground truth

    Args:
        predictions: (3, 30, 30) predicted new burns
        ground_truth: (3, 30, 30) actual new burns
        initial_fire: (30, 30) initial fire mask at t=0
        episode_name: Name of episode for display
        output_path: Path to save HTML
    """
    # Use center of grid as map center
    # For visualization, use arbitrary lat/lon (you can adjust this)
    center_lat, center_lon = 37.5, 128.0

    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )

    # Grid resolution: 400m per cell
    cell_size_deg = 0.004  # Approximate degrees per 400m

    # Helper to get lat/lon for grid cell
    def grid_to_latlon(row, col):
        lat = center_lat - (row - 15) * cell_size_deg
        lon = center_lon + (col - 15) * cell_size_deg
        return lat, lon

    # Add initial fire (black)
    fire_cells = np.where(initial_fire > 0.5)
    if len(fire_cells[0]) > 0:
        fg_init = folium.FeatureGroup(name='Initial Fire (t=0)', show=True)
        for i in range(len(fire_cells[0])):
            row, col = fire_cells[0][i], fire_cells[1][i]
            lat, lon = grid_to_latlon(row, col)
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color='black',
                fill=True,
                fillColor='black',
                fillOpacity=0.8,
                popup=f"Initial fire<br>Row: {row}, Col: {col}"
            ).add_to(fg_init)
        fg_init.add_to(m)

    # Colors for timesteps
    colors = ['#0066FF', '#00CCFF', '#00FFCC']  # Blue shades for predictions
    gt_colors = ['#FF0000', '#FF6600', '#FFAA00']  # Red/orange shades for ground truth

    # Add predictions and ground truth for each timestep
    for t in range(3):
        # Predictions (blue)
        pred_cells = np.where(predictions[t] > 0.5)
        if len(pred_cells[0]) > 0:
            fg_pred = folium.FeatureGroup(name=f'Predicted t+{t+1}', show=True)
            for i in range(len(pred_cells[0])):
                row, col = pred_cells[0][i], pred_cells[1][i]
                lat, lon = grid_to_latlon(row, col)
                prob = predictions[t, row, col]
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=3,
                    color=colors[t],
                    fill=True,
                    fillColor=colors[t],
                    fillOpacity=0.6,
                    popup=f"Predicted t+{t+1}<br>Prob: {prob:.3f}<br>Row: {row}, Col: {col}"
                ).add_to(fg_pred)
            fg_pred.add_to(m)

        # Ground truth (red/orange)
        gt_cells = np.where(ground_truth[t] > 0.5)
        if len(gt_cells[0]) > 0:
            fg_gt = folium.FeatureGroup(name=f'Ground Truth t+{t+1}', show=True)
            for i in range(len(gt_cells[0])):
                row, col = gt_cells[0][i], gt_cells[1][i]
                lat, lon = grid_to_latlon(row, col)
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=3,
                    color=gt_colors[t],
                    fill=True,
                    fillColor=gt_colors[t],
                    fillOpacity=0.7,
                    popup=f"Ground Truth t+{t+1}<br>Row: {row}, Col: {col}"
                ).add_to(fg_gt)
            fg_gt.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add legend
    legend_html = f'''
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 220px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:12px; padding: 10px">
    <p style="margin-bottom:5px;"><b>{episode_name}</b></p>
    <p style="margin:2px;"><span style="color:black">&#9632;</span> Initial Fire (t=0)</p>
    <p style="margin:2px;"><b>Predictions:</b></p>
    <p style="margin:2px;"><span style="color:{colors[0]}">&#9632;</span> t+1 Predicted</p>
    <p style="margin:2px;"><span style="color:{colors[1]}">&#9632;</span> t+2 Predicted</p>
    <p style="margin:2px;"><span style="color:{colors[2]}">&#9632;</span> t+3 Predicted</p>
    <p style="margin:2px;"><b>Ground Truth:</b></p>
    <p style="margin:2px;"><span style="color:{gt_colors[0]}">&#9632;</span> t+1 Actual</p>
    <p style="margin:2px;"><span style="color:{gt_colors[1]}">&#9632;</span> t+2 Actual</p>
    <p style="margin:2px;"><span style="color:{gt_colors[2]}">&#9632;</span> t+3 Actual</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save map
    m.save(output_path)
    print(f"Visualization saved to {output_path}")


def main():
    """Run inference on random validation episodes"""
    import argparse

    parser = argparse.ArgumentParser(description='U-Net V2 Inference')
    parser.add_argument('--checkpoint', type=str,
                       default='sl_training/unet_16ch_v2/checkpoints/run2_normalized(0.1966)/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str,
                       default='embedded_data/fire_episodes_16ch_normalized',
                       help='Directory with embedded episodes')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of validation episodes to visualize')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for inference (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='inference/sl/outputs',
                       help='Directory to save visualizations')

    args = parser.parse_args()

    # Setup paths
    root_dir = Path(__file__).parent.parent.parent  # inference/sl/unet_inference.py -> project root
    checkpoint_path = root_dir / args.checkpoint
    data_dir = root_dir / args.data_dir
    output_dir = root_dir / args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load model
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = load_model(checkpoint_path, device=device)

    # Get validation episodes (last 20%)
    all_episodes = sorted(data_dir.glob('episode_*.npz'))
    n_train = int(0.8 * len(all_episodes))
    val_episodes = all_episodes[n_train:]

    print(f"\nFound {len(val_episodes)} validation episodes")
    print(f"Running inference on {args.num_samples} random episodes...\n")

    # Sample random episodes
    sampled_episodes = random.sample(val_episodes, min(args.num_samples, len(val_episodes)))

    # Run inference on each
    for i, ep_path in enumerate(sampled_episodes):
        print("=" * 80)
        print(f"Episode {i+1}/{len(sampled_episodes)}: {ep_path.name}")
        print("=" * 80)

        # Load episode
        states, fire_masks = load_episode(ep_path)
        T = len(states)
        print(f"Episode length: {T} timesteps (MEL: {T-1})")

        # Pick a random starting timestep (need at least 3 future timesteps)
        if T <= 3:
            print("Episode too short, skipping...")
            continue

        start_t = random.randint(0, T - 4)
        print(f"Starting from timestep {start_t}")

        # Run prediction
        predictions, ground_truth, initial_fire = predict_episode(
            model, states, fire_masks, start_t, device=device
        )

        # Compute IoU (both strict and relaxed)
        ious_strict = compute_iou(predictions, ground_truth)
        ious_relaxed = compute_relaxed_iou(predictions, ground_truth)

        print(f"\nStrict IoU (exact cell match):")
        for t, iou in enumerate(ious_strict):
            print(f"  t+{t+1}: {iou:.4f} ({iou*100:.2f}%)")
        mean_strict = np.mean(ious_strict)
        print(f"  Mean: {mean_strict:.4f} ({mean_strict*100:.2f}%)")

        print(f"\nRelaxed IoU (8-neighbor tolerance):")
        for t, iou in enumerate(ious_relaxed):
            print(f"  t+{t+1}: {iou:.4f} ({iou*100:.2f}%)")
        mean_relaxed = np.mean(ious_relaxed)
        print(f"  Mean: {mean_relaxed:.4f} ({mean_relaxed*100:.2f}%)")

        # Print statistics
        print(f"\nPrediction statistics:")
        for t in range(3):
            pred_count = (predictions[t] > 0.5).sum()
            gt_count = (ground_truth[t] > 0.5).sum()
            print(f"  t+{t+1}: Predicted {pred_count} cells, Actual {gt_count} cells")

        # Create visualization
        output_path = output_dir / f"unet_v2_{ep_path.stem}_t{start_t}.html"
        create_visualization(
            predictions, ground_truth, initial_fire,
            f"{ep_path.stem} (t={start_t})",
            str(output_path)
        )
        print(f"\nVisualization: {output_path}")
        print()


if __name__ == '__main__':
    main()
