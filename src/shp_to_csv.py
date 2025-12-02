import geopandas as gpd
import glob
from config import *
from macros import *

def fetch_shpfile(yearly_path)->list[str]:
    """ Identify and fetch shapefile paths from given directory path """
    shp_pattern = os.path.join(yearly_path, "*.shp")
    shp_files = glob.glob(shp_pattern)
    recursive_pattern = os.path.join(yearly_path, "**", "*.shp")
    shp_files.extend(glob.glob(recursive_pattern, recursive=True))
    return shp_files

def save_to_csv(shp_path)->None:
    out_path = shp_path.replace(".shp", ".csv")
    gdf = gpd.read_file(shp_path)
    gdf.to_csv(out_path, index=False)

if __name__=="__main__":

    PATH = NASA_VIIRS_DATA_DIR

    yearly_paths = read_paths(PATH)
    for path in yearly_paths:
        shp_files = fetch_shpfile(path)
        save_to_csv(shp_files[0])