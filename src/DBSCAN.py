import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from config import *
from macros import *

def haversine_distance(lat1, lon1, lat2, lon2)->float:
    """Calculate haversine distance (km) between two points """
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def clustering_all_timeframes(csv_path: str, eps_km: float = 2.0, min_samples: int = 3)->None:
    """
    Run DBSCAN on a VIIRS CSV using latitude/longitude with haversine distance.
    Save clustered CSV with 'cluster' column added.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(e)
        return

    coords = np.radians(df[['LATITUDE', 'LONGITUDE']].to_numpy())
    kms_per_radian = 6371.0088
    db = DBSCAN(eps=eps_km / kms_per_radian,
                min_samples=min_samples,
                algorithm='ball_tree',
                metric='haversine')
    db.fit(coords)
    df['cluster'] = db.labels_

    base, file_name = os.path.split(csv_path)
    save_path = os.path.join(base, f"clustered_{file_name}")
    df.to_csv(save_path, index=False)
    
if __name__ == "__main__":
    paths = read_paths(NASA_VIIRS_DATA_DIR)
    for path in paths:
        for file_name in os.listdir(path):
            if file_name.endswith(".csv"):
                full_csv_path = os.path.join(path, file_name)
                clustering_all_timeframes(full_csv_path)