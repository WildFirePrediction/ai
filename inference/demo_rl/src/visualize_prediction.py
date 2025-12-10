"""
Visualize prediction results as Folium HTML map
Creates interactive map showing initial fire and predictions
"""
import json
from pathlib import Path
import argparse
import folium
import numpy as np


def create_prediction_map(prediction_json, output_file):
    """
    Create Folium map from prediction JSON

    Args:
        prediction_json: Dict with prediction results
        output_file: Path to save HTML map

    Returns:
        map_path: Path to saved map
    """
    fire_id = prediction_json['fire_id']
    fire_lat = prediction_json['fire_location']['lat']
    fire_lon = prediction_json['fire_location']['lon']
    fire_timestamp = prediction_json['fire_timestamp']
    predictions = prediction_json['predictions']

    print(f"\nCreating visualization for Fire {fire_id}")
    print(f"  Location: ({fire_lat:.4f}, {fire_lon:.4f})")
    print(f"  Time: {fire_timestamp}")

    # Create map centered at fire location
    m = folium.Map(
        location=[fire_lat, fire_lon],
        zoom_start=13,
        tiles='OpenStreetMap'
    )

    # Add initial fire point (large black marker)
    folium.Marker(
        location=[fire_lat, fire_lon],
        popup=f"<b>Initial Fire</b><br>ID: {fire_id}<br>Time: {fire_timestamp}",
        tooltip="Fire Origin",
        icon=folium.Icon(color='black', icon='fire', prefix='fa')
    ).add_to(m)

    # Circle around fire origin
    folium.Circle(
        location=[fire_lat, fire_lon],
        radius=200,  # 200m radius
        color='black',
        fill=True,
        fillColor='black',
        fillOpacity=0.3,
        popup="Initial fire location"
    ).add_to(m)

    # Colors for timesteps (blue -> cyan -> green -> yellow)
    colors = ['#0066FF', '#00AAFF', '#00DDFF', '#00FFAA', '#00FF66']
    timestep_names = ['t+10 min', 't+20 min', 't+30 min', 't+40 min', 't+50 min']

    # Add predictions for each timestep
    total_cells = 0
    for t, pred in enumerate(predictions):
        timestep = pred['timestep']
        timestamp = pred['timestamp']
        predicted_cells = pred['predicted_cells']

        if len(predicted_cells) == 0:
            print(f"  Timestep {timestep}: No predictions")
            continue

        print(f"  Timestep {timestep}: {len(predicted_cells)} cells predicted")
        total_cells += len(predicted_cells)

        # Create feature group for this timestep
        fg = folium.FeatureGroup(name=timestep_names[t], show=True)

        for cell in predicted_cells:
            lat = cell['lat']
            lon = cell['lon']
            prob = cell['probability']

            # Create circle marker for predicted cell
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=colors[t],
                fill=True,
                fillColor=colors[t],
                fillOpacity=0.7,
                popup=f"<b>{timestep_names[t]}</b><br>"
                      f"Time: {timestamp}<br>"
                      f"Probability: {prob:.3f}<br>"
                      f"Location: ({lat:.4f}, {lon:.4f})",
                tooltip=f"{timestep_names[t]}: {prob:.2f}"
            ).add_to(fg)

        fg.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Add legend
    legend_html = f'''
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 250px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 15px">
    <p style="margin-bottom:10px;"><b>Fire ID: {fire_id}</b></p>
    <p style="margin:5px 0;"><b>Predictions: {total_cells} cells</b></p>
    <hr style="margin:10px 0;">
    <p style="margin:5px 0;"><span style="color:black; font-size:18px;">&#x1F525;</span> <b>Initial Fire</b></p>
    <p style="margin:5px 0;"><span style="color:{colors[0]}">&#9632;</span> {timestep_names[0]}</p>
    <p style="margin:5px 0;"><span style="color:{colors[1]}">&#9632;</span> {timestep_names[1]}</p>
    <p style="margin:5px 0;"><span style="color:{colors[2]}">&#9632;</span> {timestep_names[2]}</p>
    <p style="margin:5px 0;"><span style="color:{colors[3]}">&#9632;</span> {timestep_names[3]}</p>
    <p style="margin:5px 0;"><span style="color:{colors[4]}">&#9632;</span> {timestep_names[4]}</p>
    <hr style="margin:10px 0;">
    <p style="margin:5px 0; font-size:12px; color:#666;">
    Model: A3C RL Agent<br>
    10-minute intervals
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save map
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))

    print(f"\nMap saved to: {output_path}")

    return output_path


def create_multi_fire_map(fire_predictions, output_file):
    """
    Create Folium map with multiple fires on one map

    Args:
        fire_predictions: List of prediction dicts (each with fire_id, fire_location, predictions)
        output_file: Path to save HTML map

    Returns:
        map_path: Path to saved map
    """
    if not fire_predictions:
        print("Error: No fire predictions provided")
        return None

    print(f"\nCreating combined visualization for {len(fire_predictions)} fires")

    # Calculate center of all fires
    all_lats = [fp['fire_location']['lat'] for fp in fire_predictions]
    all_lons = [fp['fire_location']['lon'] for fp in fire_predictions]
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)

    # Create map centered at average location
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )

    # Fire colors (one color per fire)
    fire_colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen']

    # Timestep colors (lighter shades for predictions)
    timestep_colors = {
        0: '#FF6666',  # Light red
        1: '#FF8888',
        2: '#FFAAAA',
        3: '#FFCCCC',
        4: '#FFEEEE'
    }

    total_cells_all = 0

    # Process each fire
    for fire_idx, fire_pred in enumerate(fire_predictions):
        fire_id = fire_pred['fire_id']
        fire_lat = fire_pred['fire_location']['lat']
        fire_lon = fire_pred['fire_location']['lon']
        fire_timestamp = fire_pred['fire_timestamp']
        predictions = fire_pred['predictions']

        fire_color = fire_colors[fire_idx % len(fire_colors)]

        print(f"  Fire {fire_idx+1}/{len(fire_predictions)}: {fire_id}")
        print(f"    Location: ({fire_lat:.4f}, {fire_lon:.4f})")

        # Add initial fire marker
        folium.Marker(
            location=[fire_lat, fire_lon],
            popup=f"<b>Fire {fire_id}</b><br>Time: {fire_timestamp}",
            tooltip=f"Fire {fire_id}",
            icon=folium.Icon(color=fire_color, icon='fire', prefix='fa')
        ).add_to(m)

        # Circle around fire origin
        folium.Circle(
            location=[fire_lat, fire_lon],
            radius=200,
            color=fire_color,
            fill=True,
            fillColor=fire_color,
            fillOpacity=0.4,
            popup=f"Fire {fire_id} origin"
        ).add_to(m)

        # Add predictions
        fire_total_cells = 0
        for t, pred in enumerate(predictions):
            timestep = pred['timestep']
            predicted_cells = pred['predicted_cells']

            if len(predicted_cells) == 0:
                continue

            fire_total_cells += len(predicted_cells)

            # Use progressively lighter colors for later timesteps
            color = timestep_colors.get(t, '#FFE0E0')

            for cell in predicted_cells:
                lat = cell['lat']
                lon = cell['lon']
                prob = cell['probability']

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=3,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.6,
                    popup=f"<b>Fire {fire_id} - t+{timestep}</b><br>"
                          f"Probability: {prob:.3f}",
                    tooltip=f"Fire {fire_id}"
                ).add_to(m)

        print(f"    Predictions: {fire_total_cells} cells")
        total_cells_all += fire_total_cells

    # Add legend
    legend_html = f'''
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 280px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 15px">
    <p style="margin-bottom:10px;"><b>Multi-Fire Visualization</b></p>
    <p style="margin:5px 0;"><b>{len(fire_predictions)} Fires</b></p>
    <p style="margin:5px 0;"><b>Total Predictions: {total_cells_all} cells</b></p>
    <hr style="margin:10px 0;">
    '''

    for fire_idx, fire_pred in enumerate(fire_predictions):
        fire_id = fire_pred['fire_id']
        fire_color = fire_colors[fire_idx % len(fire_colors)]
        legend_html += f'<p style="margin:5px 0;"><span style="color:{fire_color}; font-size:18px;">&#x1F525;</span> <b>{fire_id}</b></p>'

    legend_html += '''
    <hr style="margin:10px 0;">
    <p style="margin:5px 0; font-size:12px; color:#666;">
    Model: A3C RL Agent<br>
    10-minute intervals
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save map
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))

    print(f"\nCombined map saved to: {output_path}")
    print(f"Total fires: {len(fire_predictions)}")
    print(f"Total predicted cells: {total_cells_all}")

    return output_path


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Visualize prediction results as Folium map'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to prediction JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save HTML map (default: inference/demo/output/<fire_id>.html)'
    )

    args = parser.parse_args()

    # Load prediction JSON
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        return

    print(f"Loading prediction from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        prediction_json = json.load(f)

    # Determine output path
    if args.output:
        output_file = Path(args.output)
    else:
        fire_id = prediction_json['fire_id']
        output_file = Path('inference/demo/output') / f"fire_{fire_id}.html"

    # Create visualization
    create_prediction_map(prediction_json, output_file)

    print(f"\nVisualization complete!")
    print(f"Open the HTML file in a browser to view the map.")


if __name__ == '__main__':
    main()
