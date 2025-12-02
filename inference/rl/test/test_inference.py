"""
Test script for inference pipeline
Simulates real-time fire detection and prediction with visualization
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import folium
from folium import plugins
from datetime import datetime
import json

from inference.inference_engine import InferenceEngine
from inference.test.dummy_kfs_data import generate_dummy_kfs_trigger, generate_progressive_fire_triggers


def create_visualization_map(predictions, trigger_info, output_path='inference_prediction_map.html'):
    """
    Create interactive Folium map showing fire spread predictions.
    
    Args:
        predictions: List of prediction dicts from InferenceEngine
        trigger_info: Original fire trigger info
        output_path: Path to save HTML map
    """
    # Convert EPSG:5179 to lat/lon (rough approximation for visualization)
    # In production, use proper pyproj transformation
    def epsg5179_to_latlon(x, y):
        lat = 38.0 + (y - 1800000) / 111000
        lon = 127.5 + (x - 1000000) / 88800
        return lat, lon
    
    # Get initial fire location
    init_lat, init_lon = epsg5179_to_latlon(
        trigger_info['_debug']['x_epsg5179'],
        trigger_info['_debug']['y_epsg5179']
    )
    
    # Create map centered on fire
    m = folium.Map(
        location=[init_lat, init_lon],
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # Add initial fire trigger
    folium.Marker(
        location=[init_lat, init_lon],
        popup=f"Fire Trigger<br>{trigger_info['timestamp']}",
        icon=folium.Icon(color='red', icon='fire', prefix='fa'),
        tooltip='Initial Fire Detection'
    ).add_to(m)
    
    # Color scale for timesteps (blue to red)
    colors = [
        '#0000FF', '#1E00FF', '#3C00FF', '#5A00FF', '#7800FF',
        '#9600FF', '#B400FF', '#D200FF', '#F000FF', '#FF00DC'
    ]
    
    # Add predictions for each timestep
    for pred in predictions:
        timestep = pred['timestep']
        fire_coords = pred['fire_coords']
        color = colors[min(timestep-1, len(colors)-1)]
        
        # Create feature group for this timestep
        fg = folium.FeatureGroup(name=f'Timestep {timestep} ({pred["timestamp"][:16]})')
        
        for coord in fire_coords:
            lat, lon = epsg5179_to_latlon(coord['x_epsg5179'], coord['y_epsg5179'])
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=3,
                popup=f"T+{timestep}<br>{coord['timestamp'][:16]}",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6,
                weight=1
            ).add_to(fg)
        
        fg.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px; height: 280px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 10px">
    <p style="margin-bottom:5px;"><b>Fire Spread Prediction</b></p>
    <p style="margin:2px;">Initial Fire: <i class="fa fa-fire" style="color:red"></i></p>
    <p style="margin:2px;">Timesteps:</p>
    '''
    
    for i in range(min(10, len(predictions))):
        color = colors[i]
        legend_html += f'<p style="margin:2px;"><span style="color:{color}">&#9632;</span> T+{i+1}</p>'
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    m.save(output_path)
    print(f"[Visualization] Map saved to {output_path}")
    
    return m


def print_prediction_summary(predictions, trigger_info):
    """Print human-readable summary of predictions."""
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    
    print(f"\nInitial Fire Trigger:")
    print(f"  Location: {trigger_info['lat']:.6f}N, {trigger_info['lon']:.6f}E")
    print(f"  Time: {trigger_info['timestamp']}")
    print(f"  Debug Info:")
    print(f"    Global Grid: (row={trigger_info['_debug']['global_row']}, col={trigger_info['_debug']['global_col']})")
    print(f"    EPSG:5179: ({trigger_info['_debug']['x_epsg5179']:.1f}, {trigger_info['_debug']['y_epsg5179']:.1f})")
    
    print(f"\nPredicted Fire Spread:")
    total_cells = 0
    
    for pred in predictions:
        num_cells = pred['num_cells']
        total_cells += num_cells
        print(f"  Timestep {pred['timestep']:2d} ({pred['timestamp'][:16]}): {num_cells:4d} new cells burning")
    
    print(f"\nTotal cells predicted to burn: {total_cells}")
    print(f"Approximate area: {total_cells * 0.16:.2f} km² (at 400m resolution)")


def test_single_prediction():
    """Test single fire trigger and prediction."""
    print("=" * 80)
    print("TEST: Single Fire Prediction")
    print("=" * 80)
    
    # Setup paths
    root_dir = Path(__file__).parent.parent.parent
    model_path = root_dir / 'rl_training' / 'a3c' / 'checkpoints_v3r2' / 'mel4-w4-251126-1812' / 'best_model.pt'
    grid_meta_path = root_dir / 'embedded_data' / 'grid_metadata.json'
    static_data_path = root_dir / 'embedded_data' / 'state_vectors.npz'
    
    # Initialize engine
    engine = InferenceEngine(
        model_path=str(model_path),
        grid_metadata_path=str(grid_meta_path),
        static_data_path=str(static_data_path),
        device='cuda' if sys.argv[1:] and sys.argv[1] == '--gpu' else 'cpu',
        initial_grid_size=30,
        expansion_margin=10,
        expansion_size=5
    )
    
    # Generate dummy fire trigger
    print("\n[Test] Generating dummy fire trigger...")
    trigger = generate_dummy_kfs_trigger()
    
    # Dummy weather data: [temp, humidity, wind_speed, wind_x, wind_y, rainfall]
    weather = np.array([25.0, 60.0, 5.0, 2.0, 3.0, 0.0], dtype=np.float32)
    
    # Initialize from trigger
    print("\n[Test] Initializing environment from trigger...")
    engine.initialize_from_trigger(
        fire_x=trigger['_debug']['x_epsg5179'],
        fire_y=trigger['_debug']['y_epsg5179'],
        weather_data=weather,
        timestamp=trigger['timestamp_iso']
    )
    
    # Predict 10 timesteps
    print("\n[Test] Predicting 10 timesteps ahead...")
    predictions = engine.predict_timesteps(num_timesteps=10)
    
    # Print summary
    print_prediction_summary(predictions, trigger)
    
    # Create visualization
    print("\n[Test] Creating visualization map...")
    output_path = root_dir / 'inference' / 'test' / 'test_single_prediction.html'
    create_visualization_map(predictions, trigger, str(output_path))
    
    print("\n[Test] COMPLETE")
    print(f"[Test] Open {output_path} in browser to view results")


def test_hard_reset():
    """Test hard reset with multiple fire triggers."""
    print("=" * 80)
    print("TEST: Hard Reset with Multiple Triggers")
    print("=" * 80)
    
    # Setup paths
    root_dir = Path(__file__).parent.parent.parent
    model_path = root_dir / 'rl_training' / 'a3c' / 'checkpoints_v3r2' / 'mel4-w4-251126-1812' / 'best_model.pt'
    grid_meta_path = root_dir / 'embedded_data' / 'grid_metadata.json'
    static_data_path = root_dir / 'embedded_data' / 'state_vectors.npz'
    
    # Initialize engine
    engine = InferenceEngine(
        model_path=str(model_path),
        grid_metadata_path=str(grid_meta_path),
        static_data_path=str(static_data_path),
        device='cuda' if sys.argv[1:] and sys.argv[1] == '--gpu' else 'cpu'
    )
    
    # Generate multiple triggers
    print("\n[Test] Generating 3 progressive fire triggers...")
    triggers = generate_progressive_fire_triggers(num_triggers=3, interval_hours=2)
    weather = np.array([25.0, 60.0, 5.0, 2.0, 3.0, 0.0], dtype=np.float32)
    
    all_predictions = []
    
    for i, trigger in enumerate(triggers):
        print(f"\n{'=' * 80}")
        print(f"TRIGGER {i+1}/3: {trigger['timestamp']}")
        print(f"Location: {trigger['lat']:.6f}N, {trigger['lon']:.6f}E")
        print('=' * 80)
        
        # Hard reset for new trigger
        if i == 0:
            engine.initialize_from_trigger(
                fire_x=triggers[0]['_debug']['x_epsg5179'] if '_debug' in triggers[0] else trigger['lon'] * 88800 + 1000000,
                fire_y=triggers[0]['_debug']['y_epsg5179'] if '_debug' in triggers[0] else (trigger['lat'] - 38.0) * 111000 + 1800000,
                weather_data=weather,
                timestamp=trigger['timestamp_iso']
            )
        else:
            # Approximate EPSG:5179 from lat/lon for subsequent triggers
            x = (trigger['lon'] - 127.5) * 88800 + 1000000
            y = (trigger['lat'] - 38.0) * 111000 + 1800000
            engine.hard_reset(x, y, weather, trigger['timestamp_iso'])
        
        # Predict 5 timesteps for each trigger
        print(f"\n[Test] Predicting 5 timesteps from trigger {i+1}...")
        predictions = engine.predict_timesteps(num_timesteps=5)
        all_predictions.append({'trigger': trigger, 'predictions': predictions})
        
        # Print summary
        print(f"\n[Trigger {i+1}] Predicted {sum(p['num_cells'] for p in predictions)} total cells")
    
    print("\n[Test] COMPLETE - Hard reset test finished")
    print("[Test] Multiple triggers successfully processed")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test wildfire inference pipeline')
    parser.add_argument('--test', choices=['single', 'reset', 'both'], default='single',
                        help='Test to run: single prediction, hard reset, or both')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    
    args = parser.parse_args()
    
    if args.test in ['single', 'both']:
        test_single_prediction()
    
    if args.test in ['reset', 'both']:
        print("\n\n")
        test_hard_reset()
