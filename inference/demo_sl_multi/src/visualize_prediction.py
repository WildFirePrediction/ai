"""
Visualize multi-point prediction results as Folium HTML map
Shows initial fire (1 center + 2 spread points) and predictions
"""
import json
from pathlib import Path
import argparse
import folium


def create_prediction_map(prediction_json, output_file):
    """
    Create Folium map from multi-point prediction JSON

    Args:
        prediction_json: Dict with prediction results (includes spread_points)
        output_file: Path to save HTML map

    Returns:
        map_path: Path to saved map
    """
    fire_id = prediction_json['fire_id']
    fire_lat = prediction_json['fire_location']['lat']
    fire_lon = prediction_json['fire_location']['lon']
    fire_timestamp = prediction_json['fire_timestamp']
    spread_points = prediction_json.get('spread_points', [])
    total_initial = prediction_json.get('total_initial_cells', 3)
    predictions = prediction_json['predictions']

    print(f"\nCreating visualization for Fire {fire_id}")
    print(f"  Center: ({fire_lat:.4f}, {fire_lon:.4f})")
    print(f"  Spread points: {len(spread_points)}")
    print(f"  Total initial cells: {total_initial}")
    print(f"  Time: {fire_timestamp}")

    # Create map centered at fire location
    m = folium.Map(
        location=[fire_lat, fire_lon],
        zoom_start=14,
        tiles='OpenStreetMap'
    )

    # Add initial center fire point (large black marker)
    folium.Marker(
        location=[fire_lat, fire_lon],
        popup=f"<b>Initial Fire (Center)</b><br>ID: {fire_id}<br>Time: {fire_timestamp}",
        tooltip="Fire Origin",
        icon=folium.Icon(color='black', icon='fire', prefix='fa')
    ).add_to(m)

    # Circle around center fire point
    folium.Circle(
        location=[fire_lat, fire_lon],
        radius=200,  # 200m radius
        color='black',
        fill=True,
        fillColor='black',
        fillOpacity=0.4,
        popup="Initial fire center"
    ).add_to(m)

    # Add spread points (red markers)
    for i, sp in enumerate(spread_points):
        sp_lat = sp['lat']
        sp_lon = sp['lon']

        folium.CircleMarker(
            location=[sp_lat, sp_lon],
            radius=8,
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.6,
            popup=f"<b>Spread Point {i+1}</b><br>"
                  f"Already burning (t=0)<br>"
                  f"Location: ({sp_lat:.4f}, {sp_lon:.4f})",
            tooltip=f"Spread Point {i+1}"
        ).add_to(m)

    # Colors for predicted timesteps (blue -> cyan -> green -> yellow -> orange)
    colors = ['#0066FF', '#00AAFF', '#00DDFF', '#00FFAA', '#FFD700']
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
        if t < len(timestep_names):
            fg = folium.FeatureGroup(name=timestep_names[t], show=True)
        else:
            fg = folium.FeatureGroup(name=f"t+{timestep*10} min", show=True)

        for cell in predicted_cells:
            lat = cell['lat']
            lon = cell['lon']
            prob = cell['probability']

            # Create circle marker for predicted cell
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=colors[min(t, len(colors)-1)],
                fill=True,
                fillColor=colors[min(t, len(colors)-1)],
                fillOpacity=0.7,
                popup=f"<b>{timestep_names[t] if t < len(timestep_names) else f't+{timestep*10} min'}</b><br>"
                      f"Time: {timestamp}<br>"
                      f"Probability: {prob:.3f}<br>"
                      f"Location: ({lat:.4f}, {lon:.4f})",
                tooltip=f"{timestep_names[t] if t < len(timestep_names) else f't+{timestep*10} min'}: {prob:.2f}"
            ).add_to(fg)

        fg.add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Build legend entries dynamically
    legend_timesteps = ""
    for t in range(min(len(predictions), len(timestep_names))):
        legend_timesteps += f'<p style="margin:5px 0;"><span style="color:{colors[t]}">&#9632;</span> {timestep_names[t]}</p>\n'

    # Add legend
    legend_html = f'''
    <div style="position: fixed;
                bottom: 50px; right: 50px; width: 270px;
                background-color: white; border:2px solid grey; z-index:9999;
                font-size:14px; padding: 15px">
    <p style="margin-bottom:10px;"><b>Fire ID: {fire_id}</b></p>
    <p style="margin:5px 0;"><b>Initial: {total_initial} cells</b></p>
    <p style="margin:5px 0;"><b>Predictions: {total_cells} cells</b></p>
    <hr style="margin:10px 0;">
    <p style="margin:5px 0;"><span style="color:black; font-size:18px;">&#x1F525;</span> <b>Initial Fire (Center)</b></p>
    <p style="margin:5px 0;"><span style="color:red; font-size:14px;">&#9679;</span> <b>Spread Points (t=0)</b></p>
    <hr style="margin:10px 0;">
    {legend_timesteps}
    <hr style="margin:10px 0;">
    <p style="margin:5px 0; font-size:12px; color:#666;">
    Model: U-Net V3<br>
    Multi-point initialization<br>
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


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Visualize multi-point prediction results as Folium map'
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
        help='Path to save HTML map'
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
        output_file = Path('inference/demo_sl_multi/output') / f"fire_{fire_id}.html"

    # Create visualization
    create_prediction_map(prediction_json, output_file)

    print(f"\nVisualization complete!")
    print(f"Open the HTML file in a browser to view the map.")


if __name__ == '__main__':
    main()
