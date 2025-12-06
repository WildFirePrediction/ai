import os
from dotenv import load_dotenv
load_dotenv()

""" Directory paths """
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

SRC_DIR = os.path.join(ROOT_DIR, "src")
DATA_DIR = os.path.join(ROOT_DIR, "data")
INFERENCE_DATA_DIR = os.path.join(ROOT_DIR, "inference_data")

NASA_DATA_DIR = os.path.join(DATA_DIR, "NASA")
NASA_MODIS_DATA_DIR = os.path.join(NASA_DATA_DIR, "MODIS")
NASA_VIIRS_DATA_DIR = os.path.join(NASA_DATA_DIR, "VIIRS")

KMA_DATA_DIR = os.path.join(DATA_DIR, "KMA")
NDVI_DATA_DIR = os.path.join(DATA_DIR, "NDVI")
KFS_DATA_DIR = os.path.join(INFERENCE_DATA_DIR, "KFS")

""" API KEYS"""
KMA_WEATHER_TOKEN = os.getenv("KMA_API_KEY")

""" KMA API """
KMA_AWS_BASE_URL = f"https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min?authKey={KMA_WEATHER_TOKEN}"
KMA_ENDPOINTS = {
    1: {"url": KMA_AWS_BASE_URL + f"?authKey={KMA_WEATHER_TOKEN}&disp=1&help=1&tm1=", "filename": "AWS", "filetype": "csv", "desc": "AWS 매분자료"},
    2: {"url": KMA_AWS_BASE_URL + f"_cloud?authKey={KMA_WEATHER_TOKEN}&disp=1&help=1&tm1=", "filename": "AWS_cloud", "filetype": "csv", "desc": "AWS 운고운량"},
    3: {"url": KMA_AWS_BASE_URL + f"_lst?authKey={KMA_WEATHER_TOKEN}&disp=1&help=1&tm1=", "filename": "AWS_temp", "filetype": "csv", "desc": "AWS 초상온도"},
    4: {"url": KMA_AWS_BASE_URL + f"_vis?authKey={KMA_WEATHER_TOKEN}&disp=1&help=1&tm1=", "filename": "AWS_vis", "filetype": "csv", "desc": "AWS 가시거리"},
}

""" KFS API """
KFS_REALTIME_URL = "https://fd.forest.go.kr/ffas/pubConn/selectPublicFireShowList.do"
KFS_WARNINGLIST_URL = "https://fd.forest.go.kr/ffas/new/getFireWarningList.do"