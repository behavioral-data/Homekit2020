import os

import pyarrow.parquet as pq
import pandas as pd

from src.utils import load_dotenv, get_logger
import src.data.constants as constants

import multiprocessing.popen_spawn_posix

import dask
dask.config.set({"distributed.comm.timeouts.connect": "60"})

import dask.dataframe as dd

load_dotenv()
logger = get_logger()

DATASET_VERSION="2020-07-15"
RAW_DATA_PATH = os.path.join("data","raw","audere","data-export",DATASET_VERSION)
PROCESSED_DATA_PATH = os.path.join("data","processed")
DEBUG_DATA_PATH = os.path.join("data","debug")

def get_raw_dataset_path(name):
    if name in constants.MTL_NAMES:
        return os.path.join(RAW_DATA_PATH,"mtl",name)        
    elif name in constants.SURVEY_NAMES:
        return os.path.join(RAW_DATA_PATH,"survey",name)
    elif name in constants.ACTIVITY_NAMES:
        return os.path.join(RAW_DATA_PATH,"activity",name)
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
    if os.environ.get("DEBUG_DATA"): 
        logger.warning("DEBUG_DATA is set, only loading subset of data")
        data_path = DEBUG_DATA_PATH
    else:
        data_path = PROCESSED_DATA_PATH
    if name in constants.PROCESSED_DATASETS:
        return os.path.join(data_path,name+".csv")        
    elif name in constants.PARQUET_DATASETS:
        return os.path.join(data_path,name) 
    else:
        raise ValueError(f"Looked for {name} in {data_path}, not found!")
        
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
            dataset[column] = pd.to_datetime(dataset[column])
    logger.info(f"Reading {name}...")
    if fmt=="df":
        return dataset
    else:
        raise ValueError("Unsupported fmt") 

# @dask.delayed
def get_dask_df(name,min_date=None,max_date=None,index=None):
    path = get_processed_dataset_path(name)
    filters = []
    # if min_date:
    #     filters.append(("date",">=",min_date))
    # if max_date:
    #     filters.append(("date","<",max_date))
        
    if filters:
        df = dd.read_parquet(path,filters=filters)
    else: 
        df = dd.read_parquet(path,index=index)
    return df
