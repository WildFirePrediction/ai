from config import *
from kma_api import get_all_weather_data, get_weather_data
from macros import *
from read_csv import *


def filter_kma():
    paths = read_paths(KMA_DATA_DIR)
    for path in paths:
        csvs = read_files(path)
        for csv in csvs:
            filter_kma_csv(csv)


if __name__ == "__main__":
    filter_kma()
    # get_all_weather_data("202001010900")
