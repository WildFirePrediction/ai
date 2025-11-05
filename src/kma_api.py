import csv
import os
import time

import numpy as np
import requests

from config import KMA_DATA_DIR, KMA_AWS_BASE_URL

np.seterr(invalid="ignore", divide="ignore")


def get_weather_data(endpoint, out_dir, timestamp):
    url = endpoint
    url += f"&tm2={timestamp}"
    filename = "AWS"
    filetype = "csv"

    try:
        response = requests.get(url)
    except Exception as e:
        print(e)
        return

    os.makedirs(out_dir, exist_ok=True)

    temp_path = os.path.join(out_dir, f"{filename}.{filetype}")
    save_path = os.path.join(out_dir, f"{filename}_{timestamp}.{filetype}")

    if filetype == "csv":
        response.encoding = 'euc-kr'
        with open(temp_path, "w", encoding="utf-8") as f:
            f.write(response.text)

    convert_to_csv(temp_path, save_path)
    print(f"saved to : {save_path}")


def get_all_weather_data(timestamp):
    out_dir = os.path.join(KMA_DATA_DIR, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    get_weather_data(KMA_AWS_BASE_URL, out_dir, timestamp)


def convert_to_csv(infile_path, outfile_path):
    with open(infile_path, "r", encoding="utf-8") as f_in:
        lines = f_in.readlines()

    data_start = None
    for i, line in enumerate(lines):
        if line.strip() == "":
            continue
        if not line.lstrip().startswith("#"):
            data_start = i
            break
    if data_start is None:
        raise ValueError("No data line")

    header_line = None
    for line in lines:
        if line.startswith("#") and "YYMMDDHHMI" in line:
            header_line = line.strip()
            break
    if header_line is None:
        raise Exception("No header line")

    raw_headers = header_line.lstrip("#").strip().split()

    def clean_header(tok: str) -> str:
        while tok.endswith("."):
            tok = tok[:-1]
        return tok

    headers = [clean_header(h) for h in raw_headers]

    if headers and headers[0].upper().startswith("YYMMDD"):
        headers[0] = "TIME"

    data_rows = []
    for line in lines[data_start:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.strip().split()
        data_rows.append(parts)

    with open(outfile_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(headers)
        for row in data_rows:
            writer.writerow(row)

    os.remove(infile_path)

if __name__ == "__main__":
    get_all_weather_data("201202010340")