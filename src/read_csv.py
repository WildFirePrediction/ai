import csv
from pathlib import Path

import pandas as pd

from config import *
from macros import *
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


def filter_kma_csv(file_path):
    """
    Extract : wind direction, wind speed, temp, Rain event(+rain amount), humidity

    """
    file_path = Path(file_path)
    save_path = file_path.parent / "filtered_dataset.csv"
    df = pd.read_csv(file_path)
    subset = df[["WD1", "WS1", "WDS", "WSS", "WD10", "WS10",
                 "TA", "RE", "RN-15m", "RN-60m", "RN-12H", "RN-DAY", "HM"]]
    subset.to_csv(save_path, index=False)
