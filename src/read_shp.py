import geopandas as gpd
import matplotlib.pyplot as plt
import os
import glob
from config import NASA_MODIS_DATA_DIR, NASA_VIIRS_DATA_DIR

def fetch_shpfile(yearly_path)->list[str]:
    """ Identify and fetch shapefile paths from given directory path """
    shp_pattern = os.path.join(yearly_path, "*.shp")
    shp_files = glob.glob(shp_pattern)
    recursive_pattern = os.path.join(yearly_path, "**", "*.shp")
    shp_files.extend(glob.glob(recursive_pattern, recursive=True))
    return shp_files

def read_paths(data_path)->list[str]:
    """ Read available directories and return a list of paths """
    if not os.path.exists(data_path):
        return []
    directories = [
        os.path.join(data_path, item)
        for item in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, item))
    ]
    return directories

def save_to_csv(shp_path)->None:
    out_path = shp_path.replace(".shp", ".csv")
    gdf = gpd.read_file(shp_path)
    gdf.to_csv(out_path, index=False)

if __name__=="__main__":
    yearly_paths = read_paths(NASA_VIIRS_DATA_DIR)
    for path in yearly_paths:
        shp_files = fetch_shpfile(path)

        #gdf = gpd.read_file(shp_files[0])

        save_to_csv(shp_files[0])
        """

        plt.figure(figsize=(12, 12))
        gdf.plot(facecolor='lightblue', edgecolor='black')
        plt.title("고화질 시각화")
        plt.show()
        for shp in shp_files:
            gdf = gpd.read_file(shp)

            print(gdf.head())
            print(gdf.columns)
            print(gdf.crs)

            gdf.plot()
            plt.title("Shapefile")
            plt.show()
        """