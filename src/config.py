import os
""" Directory paths """

ROOT_DIR = os.path.dirname(os.getcwd())
SRC_DIR = os.path.join(ROOT_DIR, "src")
DATA_DIR = os.path.join(ROOT_DIR, "data")
NASA_DATA_DIR = os.path.join(DATA_DIR, "NASA")
NASA_MODIS_DATA_DIR = os.path.join(NASA_DATA_DIR, "MODIS")
NASA_VIIRS_DATA_DIR = os.path.join(NASA_DATA_DIR, "VIIRS")
MAP_DATA_DIR = os.path.join(ROOT_DIR, "map")