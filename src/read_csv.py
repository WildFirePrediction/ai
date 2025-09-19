import csv
import os
from config import NASA_DATA_DIR, NASA_VIIRS_DATA_DIR, NASA_MODIS_DATA_DIR

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
