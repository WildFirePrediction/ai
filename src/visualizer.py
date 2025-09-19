import folium
import os
from config import MAP_DATA_DIR

def draw_fire_grid():
    save_path = os.path.join(MAP_DATA_DIR, "fire_map_firms.html")
    grid =[ ]
    if not grid:
        return
    center_lat = sum(d['LAT'] for d in grid) / len(grid)
    center_lon = sum(d['LON'] for d in grid) / len(grid)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)
    for d in grid:
        if d.get("WARNING_LEVEL") == "주의보":
            icon_color = "orange"
        elif d.get("WARNING_LEVEL") == "경보":
            icon_color = "red"
        else: icon_color = "green"
        popup_text = (
            f"발생일시 : {d.get('DATE')}\n"
            f"주소 : {d.get('LOCATION')}\n"
            f"단계 : {d.get('WARNING_LEVEL')}"
        )
        folium.Marker(
            location=[d.get("LAT"), d.get("LON")],
            popup=popup_text,
            icon=folium.Icon(color=icon_color, icon="info-sign")
        ).add_to(m)

    m.save(save_path)
