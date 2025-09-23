import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
from config import *
from macros import *

def cluster_big_fires(df, eps_km=2.0, min_samples=3)->pd.DataFrame:
    """
    DBSCAN clustering on big fires using LATITUDE/LONGITUDE
    """
    coords = np.radians(df[['LATITUDE', 'LONGITUDE']].to_numpy())
    kms_per_radian = 6371.0088
    db = DBSCAN(eps=eps_km / kms_per_radian,
                min_samples=min_samples,
                algorithm='ball_tree',
                metric='haversine')
    db.fit(coords)
    df['cluster'] = db.labels_
    return df

def chop_clusters_by_time(df, time_interval_minutes=60, save_base_dir=None, folder_name="")->None:
    """
    Chop clustered big fires by time intervals and save CSVs
    """
    if save_base_dir is None:
        save_base_dir = os.path.join(os.getcwd(), "chopped")

    clusters = df['cluster'].unique()
    for cluster_id in clusters:
        if cluster_id == -1:
            continue
        cluster_df = df[df['cluster'] == cluster_id]

        cluster_df.loc[:, 'ACQ_DATETIME'] = pd.to_datetime(
            cluster_df['ACQ_DATE'].astype(str) +
            cluster_df['ACQ_TIME'].astype(str).str.zfill(4),
            format='%Y-%m-%d%H%M'
        )

        start_time = cluster_df['ACQ_DATETIME'].min()
        end_time = cluster_df['ACQ_DATETIME'].max()
        interval = timedelta(minutes=time_interval_minutes)

        chopped_dir = os.path.join(save_base_dir, folder_name, f"cluster{cluster_id}")
        os.makedirs(chopped_dir, exist_ok=True)

        current_start = start_time
        while current_start <= end_time:
            current_end = current_start + interval
            timeframe_df = cluster_df[
                (cluster_df['ACQ_DATETIME'] >= current_start) &
                (cluster_df['ACQ_DATETIME'] < current_end)
            ]
            if not timeframe_df.empty:
                file_name = f"cluster{cluster_id}_{current_start.strftime('%Y%m%d_%H%M')}.csv"
                save_path = os.path.join(chopped_dir, file_name)
                timeframe_df.to_csv(save_path, index=False)
            current_start = current_end

def process_viirs_folder(folder_path, eps_km=2.0, min_samples=3, time_interval_minutes=60)->None:
    """
    Process one DL_FIRE folder: merge CSVs, cluster big fires, chop by time
    """
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    dfs = []
    for f in csv_files:
        csv_path = os.path.join(folder_path, f)
        try:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        except Exception as e:
            print(e)
            continue

    if not dfs:
        return

    merged_df = pd.concat(dfs, ignore_index=True)
    clustered_df = cluster_big_fires(merged_df, eps_km=eps_km, min_samples=min_samples)
    chop_clusters_by_time(clustered_df,
                          time_interval_minutes=time_interval_minutes,
                          save_base_dir=os.path.join(folder_path, "chopped"),
                          folder_name="")

if __name__ == "__main__":
    dl_fire_dirs = read_paths(NASA_VIIRS_DATA_DIR)
    for file in dl_fire_dirs:
        print(f'Processing {file}')
        process_viirs_folder(file,
                             eps_km=2.0,
                             min_samples=3,
                             time_interval_minutes=60)