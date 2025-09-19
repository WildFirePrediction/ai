import folium
import pandas as pd
import random
from config import *
from macros import *

def visualize_clusters(csv_path: str, save_html: str = None, map_center=None, zoom_start=6):
    """
    Visualize clustered VIIRS/산불 points using Folium.

    Parameters:
        csv_path (str): Path to clustered CSV with 'latitude', 'longitude', 'cluster' columns.
        save_html (str, optional): If provided, save the map as HTML file.
        map_center (tuple, optional): (lat, lon) center of the map. If None, center on data mean.
        zoom_start (int): Initial zoom level.
    """
    df = pd.read_csv(csv_path)
    if not {'LATITUDE', 'LONGITUDE', 'cluster'}.issubset(df.columns):
        raise ValueError("CSV must contain 'latitude', 'longitude', and 'cluster' columns.")

    if map_center is None:
        map_center = (df['LATITUDE'].mean(), df['LONGITUDE'].mean())

    m = folium.Map(location=map_center, zoom_start=zoom_start)

    clusters = df['cluster'].unique()
    colors = {}
    for cluster in clusters:
        if cluster == -1:
            colors[cluster] = 'gray'
        else:
            colors[cluster] = "#%06x" % random.randint(0, 0xFFFFFF)

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=(row['LATITUDE'], row['LONGITUDE']),
            radius=3,
            color=colors[row['cluster']],
            fill=True,
            fill_opacity=0.7
        ).add_to(m)

    if save_html:
        m.save(save_html)
    return m

if __name__ == '__main__':
    paths = read_paths(NASA_VIIRS_DATA_DIR)
    for path in paths:
        for file_name in os.listdir(path):
            if file_name.startswith("clustered_") and file_name.endswith(".csv"):
                full_csv_path = os.path.join(path, file_name)
                save_html = os.path.join(path, f"{file_name.replace('.csv', '')}_map.html")

                visualize_clusters(full_csv_path, save_html=save_html)