import os

import pyarrow.parquet as pq
import pandas as pd

from src.utils import load_dotenv, get_logger
import src.data.constants as constants

load_dotenv()
logger = get_logger()

DATASET_VERSION="2020-07-15"
RAW_DATA_PATH = os.path.join("data","raw","audere","data-export",DATASET_VERSION)
PROCESSED_DATA_PATH = os.path.join("data","processed")

def get_raw_dataset_path(name):
    if name in constants.MTL_NAMES:
        return os.path.join(RAW_DATA_PATH,"mtl",name)        
    elif name in constants.SURVEY_NAMES:
        return os.path.join(RAW_DATA_PATH,"survey",name)
    else:
        raise ValueError(f"Looked for {name} in {RAW_DATA_PATH}, not found!")

def find_raw_dataset(name):
    path = get_raw_dataset_path(name)
    return pq.ParquetDataset(path)


def load_raw_table(name,fmt="df"):
    dataset = find_raw_dataset(name)
    logger.info(f"Reading {name}...")
    if fmt=="df":
        return dataset.read().to_pandas()
    elif fmt=="pq":
        return dataset.read()
    else:
        raise ValueError("Unsupported fmt") 

def get_processed_dataset_path(name):
    if name in constants.PROCESSED_DATASETS:
        return os.path.join(PROCESSED_DATA_PATH,name+".csv")        
    else:
        raise ValueError(f"Looked for {name} in {PROCESSED_DATA_PATH}, not found!")
        
def find_processed_dataset(name):
    path = get_processed_dataset_path(name)
    if ".csv" in path:
        return pd.read_csv(path)
    elif ".jsonl" in path:
        return pd.read_json(path,lines=True)

def load_processed_table(name,fmt="df"):
    dataset = find_processed_dataset(name)
    for column in dataset.columns:
        if "datetime" in str(column):
            dataset[column] = pd.to_datetime(dataset[column]).dt.tz_convert(None)

    logger.info(f"Reading {name}...")
    if fmt=="df":
        return dataset
    else:
        raise ValueError("Unsupported fmt") 
