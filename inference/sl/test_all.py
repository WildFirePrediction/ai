"""
Run inference on entire test set and save with IoU-based naming
Processes all test episodes (1979-2374) with U-Net V3
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import folium
from folium import plugins
import random

from sl_training.unet_16ch_v3.model import UNetMultiTimestep
from sl_training.unet_16ch_v3.dataset import compute_iou


def load_model(checkpoint_path, device='cuda'):
    """Load trained U-Net V3 model"""
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
    """Run inference on episode starting from timestep start_t"""
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


def compute_relaxed_iou(pred, target, threshold=0.5):
    """Compute relaxed IoU with 8-neighbor tolerance (dilated target)"""
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


def create_visualization(predictions, ground_truth, initial_fire, episode_name, mean_iou, output_path):
    """Create Folium map showing predictions vs ground truth"""
    # Use center of grid as map center
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

    # Add legend with IoU
    legend_html = f'''
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 220px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:12px; padding: 10px">
    <p style="margin-bottom:5px;"><b>{episode_name}</b></p>
    <p style="margin:2px;"><b>Relaxed IoU: {mean_iou:.4f} ({mean_iou*100:.2f}%)</b></p>
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


def main():
    """Run inference on entire test set"""
    import argparse

    parser = argparse.ArgumentParser(description='U-Net V3 Test Set Inference')
    parser.add_argument('--checkpoint', type=str,
                       default='sl_training/unet_16ch_v3/checkpoints/run1_dilated(0.3642)/best.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str,
                       default='embedded_data/fire_episodes_16ch_normalized',
                       help='Directory with embedded episodes')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for inference (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='inference/sl/test',
                       help='Directory to save visualizations')

    args = parser.parse_args()

    # Setup paths
    root_dir = Path(__file__).parent.parent.parent
    checkpoint_path = root_dir / args.checkpoint
    data_dir = root_dir / args.data_dir
    output_dir = root_dir / args.output_dir
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load model
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = load_model(checkpoint_path, device=device)

    # Get test episodes (1979-2374 = 396 episodes)
    all_episodes = sorted(data_dir.glob('episode_*.npz'))
    test_episodes = all_episodes[1979:2375]

    print(f"\n{'='*80}")
    print(f"Running inference on TEST SET")
    print(f"{'='*80}")
    print(f"Total test episodes: {len(test_episodes)}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    # Run inference on all test episodes
    results = []

    for i, ep_path in enumerate(test_episodes):
        episode_num = int(ep_path.stem.split('_')[1])

        # Load episode
        states, fire_masks = load_episode(ep_path)
        T = len(states)

        if T <= 3:
            print(f"[{i+1}/{len(test_episodes)}] {ep_path.name}: Too short (T={T}), skipping...")
            continue

        # Pick a random starting timestep (need at least 3 future timesteps)
        start_t = random.randint(0, T - 4)

        # Run prediction
        predictions, ground_truth, initial_fire = predict_episode(
            model, states, fire_masks, start_t, device=device
        )

        # Compute relaxed IoU
        ious_relaxed = compute_relaxed_iou(predictions, ground_truth)
        mean_relaxed = np.mean(ious_relaxed)

        # Create visualization with IoU-based filename
        output_filename = f"episode_{episode_num:04d}_{mean_relaxed:.4f}.html"
        output_path = output_dir / output_filename

        create_visualization(
            predictions, ground_truth, initial_fire,
            f"Episode {episode_num} (t={start_t})",
            mean_relaxed,
            str(output_path)
        )

        results.append({
            'episode': episode_num,
            'start_t': start_t,
            'mean_iou': mean_relaxed,
            'iou_t1': ious_relaxed[0],
            'iou_t2': ious_relaxed[1],
            'iou_t3': ious_relaxed[2],
        })

        # Progress update
        if (i + 1) % 10 == 0 or (i + 1) == len(test_episodes):
            avg_iou = np.mean([r['mean_iou'] for r in results])
            print(f"[{i+1}/{len(test_episodes)}] Episode {episode_num}: {mean_relaxed:.4f} | Avg so far: {avg_iou:.4f}")

    # Final summary
    print(f"\n{'='*80}")
    print(f"INFERENCE COMPLETE!")
    print(f"{'='*80}")
    print(f"Episodes processed: {len(results)}")

    mean_ious = [r['mean_iou'] for r in results]
    print(f"Mean Relaxed IoU: {np.mean(mean_ious):.4f}")
    print(f"Median Relaxed IoU: {np.median(mean_ious):.4f}")
    print(f"Std IoU: {np.std(mean_ious):.4f}")
    print(f"Min IoU: {np.min(mean_ious):.4f}")
    print(f"Max IoU: {np.max(mean_ious):.4f}")

    # Top 10
    sorted_results = sorted(results, key=lambda x: x['mean_iou'], reverse=True)
    print(f"\nTop 10 Episodes:")
    for r in sorted_results[:10]:
        print(f"  episode_{r['episode']:04d}: {r['mean_iou']:.4f}")

    print(f"\nAll visualizations saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
