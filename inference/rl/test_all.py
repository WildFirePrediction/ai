"""
Run RL inference on entire test set and save with IoU-based naming
Processes all test episodes (1979-2374) with A3C V3 LSTM REL
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import folium
from folium import plugins
import random

from rl_training.a3c_16ch.V3_LSTM_REL.model import A3C_PerCellModel_LSTM


def load_model(checkpoint_path, device='cuda'):
    """Load trained A3C model"""
    print(f"Loading A3C model from {checkpoint_path}")

    model = A3C_PerCellModel_LSTM(
        in_channels=16,
        lstm_hidden=256,
        sequence_length=3,
        use_groupnorm=True
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    best_iou = checkpoint.get('best_iou', 'N/A') if isinstance(checkpoint, dict) else 'N/A'
    episode = checkpoint.get('episode', 'N/A') if isinstance(checkpoint, dict) else 'N/A'
    print(f"Model loaded: Episode {episode}, Best IoU: {best_iou}")

    return model


def load_episode(episode_path):
    """Load a single episode from .npz file"""
    data = np.load(episode_path)
    states = data['states']  # (T, 16, 30, 30)
    fire_masks = data['fire_masks']  # (T, 30, 30)

    return states, fire_masks


def predict_episode_iterative(model, states, fire_masks, start_t, num_timesteps=5, device='cuda'):
    """
    Run RL inference on episode starting from timestep start_t
    Uses iterative prediction (fire spreads step by step)
    """
    # Get initial state
    state_t = states[start_t]  # (16, 30, 30) - environmental data
    fire_t = fire_masks[start_t]  # (30, 30) - initial fire

    # Convert to tensors
    env_tensor = torch.from_numpy(state_t).float().to(device)

    # Create temporal sequence by repeating environmental data
    sequence = env_tensor.unsqueeze(0).repeat(3, 1, 1, 1)  # (3, 16, 30, 30)
    sequence = sequence.unsqueeze(0)  # (1, 3, 16, 30, 30)

    # Initialize fire mask
    current_fire_mask = torch.from_numpy(fire_t).float().to(device)

    predictions = []

    # Iteratively predict for each timestep
    for t in range(num_timesteps):
        fire_mask_batch = current_fire_mask.unsqueeze(0)  # (1, 30, 30)

        # Run model inference
        with torch.no_grad():
            try:
                action_grid, _, _, _, _ = model.get_action_and_value(sequence, fire_mask_batch, action=None)
            except RuntimeError as e:
                # Handle numerical instability
                print(f"  WARNING: Prediction failed at t+{t+1}, stopping early")
                break

        # Update cumulative fire mask
        current_fire_mask = torch.maximum(current_fire_mask, action_grid.to(device))

        # Store cumulative fire state
        predictions.append(current_fire_mask.cpu().numpy())

    # Pad with zeros if we stopped early
    while len(predictions) < num_timesteps:
        predictions.append(np.zeros((30, 30), dtype=np.float32))

    predictions = np.stack(predictions, axis=0)  # (num_timesteps, 30, 30)

    # Convert predictions to NEW BURNS only (for IoU calculation)
    predictions_new_burns = np.zeros_like(predictions)
    predictions_new_burns[0] = (predictions[0] > 0.5) & (fire_t < 0.5)  # t+1: new cells not in initial fire
    for t in range(1, num_timesteps):
        predictions_new_burns[t] = (predictions[t] > 0.5) & (predictions[t-1] < 0.5)  # New cells vs previous

    # Get ground truth NEW BURNS
    ground_truth = []
    for dt in range(1, num_timesteps + 1):
        if start_t + dt >= len(fire_masks):
            ground_truth.append(np.zeros((30, 30), dtype=np.float32))
        else:
            fire_prev = fire_masks[start_t + dt - 1]
            fire_next = fire_masks[start_t + dt]

            actual_mask_prev = fire_prev > 0.5
            actual_mask_next = fire_next > 0.5
            new_burns = (actual_mask_next & ~actual_mask_prev).astype(np.float32)
            ground_truth.append(new_burns)

    ground_truth = np.stack(ground_truth, axis=0)  # (num_timesteps, 30, 30)

    return predictions_new_burns, ground_truth, fire_t


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


def create_visualization(predictions, ground_truth, initial_fire, episode_name, mean_iou, num_timesteps, output_path):
    """Create Folium map showing predictions vs ground truth"""
    center_lat, center_lon = 37.5, 128.0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )

    cell_size_deg = 0.004  # Approximate degrees per 400m

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

    # Colors for timesteps (blue for pred, red/orange for GT)
    colors = ['#0066FF', '#00AAFF', '#00DDFF', '#00FFDD', '#00FFAA']  # Blue shades
    gt_colors = ['#FF0000', '#FF4400', '#FF8800', '#FFCC00', '#FFFF00']  # Red/orange/yellow

    # Add predictions and ground truth for each timestep
    for t in range(num_timesteps):
        # Predictions (blue)
        pred_cells = np.where(predictions[t] > 0.5)
        if len(pred_cells[0]) > 0:
            fg_pred = folium.FeatureGroup(name=f'Predicted t+{t+1}', show=True)
            for i in range(len(pred_cells[0])):
                row, col = pred_cells[0][i], pred_cells[1][i]
                lat, lon = grid_to_latlon(row, col)
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=3,
                    color=colors[t],
                    fill=True,
                    fillColor=colors[t],
                    fillOpacity=0.6,
                    popup=f"Predicted t+{t+1}<br>Row: {row}, Col: {col}"
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
                    fillOpacity=0.6,
                    popup=f"Ground Truth t+{t+1}<br>Row: {row}, Col: {col}"
                ).add_to(fg_gt)
            fg_gt.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add title and stats
    title_html = f'''
    <div style="position: fixed;
                top: 10px; left: 50px; width: 300px; height: auto;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 10px">
    <h4 style="margin:0 0 10px 0;">{episode_name}</h4>
    <p style="margin:2px;"><b>Relaxed IoU: {mean_iou:.4f}</b></p>
    <p style="margin:2px; font-size:12px; color:#666;">A3C V3 LSTM REL (RL Agent)</p>
    <hr style="margin:5px 0;">
    <p style="margin:2px;"><span style="color:black">&#9632;</span> Initial Fire (t=0)</p>
    <p style="margin:2px;"><b>Predictions (RL):</b></p>
    '''
    for t in range(num_timesteps):
        title_html += f'<p style="margin:2px;"><span style="color:{colors[t]}">&#9632;</span> t+{t+1} Predicted</p>\n'
    title_html += '<p style="margin:2px;"><b>Ground Truth:</b></p>\n'
    for t in range(num_timesteps):
        title_html += f'<p style="margin:2px;"><span style="color:{gt_colors[t]}">&#9632;</span> t+{t+1} Actual</p>\n'
    title_html += '</div>'

    m.get_root().html.add_child(folium.Element(title_html))

    # Save map
    m.save(output_path)


def main():
    """Run RL inference on entire test set"""
    import argparse

    parser = argparse.ArgumentParser(description='A3C RL Test Set Inference')
    parser.add_argument('--checkpoint', type=str,
                       default='rl_training/a3c_16ch/V4_LSTM_REL/checkpoints/run1_relaxed/best_model.pt',
                       help='Path to A3C model checkpoint')
    parser.add_argument('--data-dir', type=str,
                       default='embedded_data/fire_episodes_16ch_normalized',
                       help='Directory with embedded episodes')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for inference (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='inference/rl/test',
                       help='Directory to save visualizations')
    parser.add_argument('--num-timesteps', type=int, default=5,
                       help='Number of timesteps to predict')

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
    print(f"Running RL inference on TEST SET")
    print(f"{'='*80}")
    print(f"Total test episodes: {len(test_episodes)}")
    print(f"Timesteps to predict: {args.num_timesteps}")
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

        if T <= args.num_timesteps:
            print(f"[{i+1}/{len(test_episodes)}] {ep_path.name}: Too short (T={T}), skipping...")
            continue

        # Pick a random starting timestep (need at least num_timesteps future timesteps)
        start_t = random.randint(0, T - args.num_timesteps - 1)

        # Run RL prediction
        predictions, ground_truth, initial_fire = predict_episode_iterative(
            model, states, fire_masks, start_t, num_timesteps=args.num_timesteps, device=device
        )

        # Compute relaxed IoU
        ious_relaxed = compute_relaxed_iou(predictions, ground_truth)
        mean_relaxed = np.mean(ious_relaxed)

        # Create visualization with IoU-based filename
        iou_percentage = round(mean_relaxed * 100)
        output_filename = f"{iou_percentage}_{episode_num:04d}.html"
        output_path = output_dir / output_filename

        create_visualization(
            predictions, ground_truth, initial_fire,
            f"Episode {episode_num} (t={start_t})",
            mean_relaxed,
            args.num_timesteps,
            str(output_path)
        )

        results.append({
            'episode': episode_num,
            'start_t': start_t,
            'mean_iou': mean_relaxed,
            'ious': ious_relaxed,
        })

        # Progress update
        if (i + 1) % 10 == 0 or (i + 1) == len(test_episodes):
            avg_iou = np.mean([r['mean_iou'] for r in results])
            print(f"[{i+1}/{len(test_episodes)}] Episode {episode_num}: {mean_relaxed:.4f} | Avg so far: {avg_iou:.4f}")

    # Final summary
    print(f"\n{'='*80}")
    print(f"RL INFERENCE COMPLETE!")
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
    print(f"Files are named: <IoU>_episode_XXXX.html (sorted by IoU)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
