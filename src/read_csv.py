import csv
import os
from config import NASA_DATA_DIR, NASA_VIIRS_DATA_DIR, NASA_MODIS_DATA_DIR
from macros import *

def read_csv(file_path: str) -> list[dict]:
    """
    Read a VIIRS CSV file and return a list of rows as dictionaries.
    Each row maps column names (header) to values.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    rows = []
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

if __name__ == "__main__":
    PATH = NASA_VIIRS_DATA_DIR
    DIRS = read_paths(PATH)

    for d in DIRS:
        for file in os.listdir(d):
            if file.endswith(".csv"):
                file_path = os.path.join(d, file)
                data = read_csv(file_path)
