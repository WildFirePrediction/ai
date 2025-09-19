import pandas as pd
from datetime import datetime, timedelta
from macros import *
from config import *

def chop_by_cluster_and_time(csv_path: str, time_interval_minutes: int = 60, save_dir: str = None):
    """
    Chop a clustered VIIRS CSV into separate CSVs by cluster and time interval.

    Parameters:
        csv_path (str): Path to the original clustered CSV ('cluster' column required).
        time_interval_minutes (int): Time interval in minutes to chop data.
        save_dir (str): Directory to save chopped CSVs. If None, saves in same folder as CSV.
    """
    df = pd.read_csv(csv_path)

    required_cols = {'LATITUDE', 'LONGITUDE', 'cluster', 'ACQ_DATE', 'ACQ_TIME'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    df['ACQ_DATETIME'] = pd.to_datetime(
        df['ACQ_DATE'].astype(str) + df['ACQ_TIME'].astype(str).str.zfill(4),
        format='%Y-%m-%d%H%M'
    )

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(csv_path), "chopped")
    os.makedirs(save_dir, exist_ok=True)

    for cluster_id in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster_id]
        if cluster_id == -1:
            continue

        start_time = cluster_df['ACQ_DATETIME'].min()
        end_time = cluster_df['ACQ_DATETIME'].max()
        interval = timedelta(minutes=time_interval_minutes)

        current_start = start_time
        while current_start <= end_time:
            current_end = current_start + interval
            timeframe_df = cluster_df[
                (cluster_df['ACQ_DATETIME'] >= current_start) &
                (cluster_df['ACQ_DATETIME'] < current_end)
            ]
            if not timeframe_df.empty:
                file_name = f"cluster{cluster_id}_{current_start.strftime('%Y%m%d_%H%M')}.csv"
                save_path = os.path.join(save_dir, file_name)
                timeframe_df.to_csv(save_path, index=False)
            current_start = current_end

if __name__ == "__main__":
    paths = read_paths(NASA_VIIRS_DATA_DIR)
    for path in paths:
        for file_name in os.listdir(path):
            if file_name.endswith(".csv"):
                full_csv_path = os.path.join(path, file_name)
                chop_by_cluster_and_time(full_csv_path, time_interval_minutes=60)