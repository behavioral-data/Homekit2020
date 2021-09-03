import os
import glob 
import json
import pickle

import pyarrow.parquet as pq
import pandas as pd
from scipy.special import softmax
import pyarrow as pa
from torch.utils import data
import numpy as np

from src.utils import get_logger
import src.data.constants as constants

import wandb
import dask
dask.config.set({"distributed.comm.timeouts.connect": "60"})

import dask.dataframe as dd

from dotenv import dotenv_values

config = dotenv_values(".env")
logger = get_logger(__name__)

DATASET_VERSION="2020-07-15"
main_path = os.getcwd()
RAW_DATA_PATH = os.path.join(main_path,"data","raw","audere","data-export",DATASET_VERSION)
PROCESSED_DATA_PATH = os.path.join(main_path,"data","processed")
DEBUG_DATA_PATH = os.path.join(main_path,"data","debug")

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

def get_features_path(name):
    if os.environ.get("DEBUG_DATA"): 
        logger.warning("DEBUG_DATA is set, only loading subset of data")
        data_path = DEBUG_DATA_PATH
    else:
        data_path = PROCESSED_DATA_PATH
    return os.path.join(data_path,"features",name+".csv")

def load_raw_table(name=None,path=None,fmt="df"):
    if not path:
        if not name:
            raise ValueError("Must provided either a known dataset name or a path")
        dataset = find_raw_dataset(name)
        logger.info(f"Reading {name}...")
    else:
        dataset = pq.ParquetDataset(path)
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

def get_cached_datareader_path(name):
    if os.environ.get("DEBUG_DATA"): 
        logger.warning("DEBUG_DATA is set, only loading subset of data")
        data_path = DEBUG_DATA_PATH
    else:
        data_path = PROCESSED_DATA_PATH
    print(data_path)
    return os.path.join(data_path,"cached_datareaders",name+".pickle")
        
def find_processed_dataset(name,path=None):
    if path is None:
        path = get_processed_dataset_path(name)
    if ".csv" in path:
        return pd.read_csv(path)
    elif ".jsonl" in path:
        return pd.read_json(path,lines=True)

def load_processed_table(name,fmt="df",path=None):
    dataset = find_processed_dataset(name,path=path)
    for column in dataset.columns:
        if "date" in str(column) or "time" in str(column):
            try:
                dataset[column] = pd.to_datetime(dataset[column])
            except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
                continue
    logger.info(f"Reading {name}...")
    if fmt=="df":
        return dataset
    else:
        raise ValueError("Unsupported fmt") 

def write_pandas_to_parquet(df,path,write_metadata=True,
                            partition_cols=[]):
    table = pa.Table.from_pandas(df, preserve_index=False)
    
    pq.write_to_dataset(table, root_path=path,
                    partition_cols=partition_cols)
    if write_metadata:
        paths = glob.glob(os.path.join(path,"*","*.parquet"))
        dd.io.parquet.create_metadata_file(paths)

def download_wandb_table(run_id,table_name="roc_table",
                 entity="mikeamerrill", project="flu"):
    api = wandb.Api()
    artifact = api.artifact(f'{entity}/{project}/run-{run_id}-{table_name}:latest')
    dir = artifact.download() 
    filenames = list(glob.glob(os.path.join(dir,"**/*.json"),recursive=True))
    data = load_json(filenames[0])
    return pd.DataFrame(data["data"],columns=data["columns"])


def get_dask_df(name=None,path= None,min_date=None,max_date=None,index=None):
    if not path:
        path = get_processed_dataset_path(name)
    filters = []
        
    if filters:
        df = dd.read_parquet(path,filters=filters)
    else: 
        df = dd.read_parquet(path,index=index)
    return df

def load_results(path):
    results = pd.read_json(path,lines=True)
    logits = pd.DataFrame(results["logits"].tolist(), columns=["pos_logit","neg_logit"])
    softmax_results = softmax(logits,axis=1)["neg_logit"].rename("pos_prob")
    return pd.concat([results["label"],logits,softmax_results],axis=1)

def write_dict_to_json(data,path,safe=True):
    if safe:
        data = {k:v for k,v in data.items() if is_jsonable(v)}

    with open(path, 'w') as outfile:
        json.dump(data, outfile)
    
def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

def load_json(path):
    with open(path, 'r') as infile:
        return json.load(infile)

def validate_reader(dataset_args,data_reader):
    props = ["min_date","max_date","split_date","min_windows",
            "day_window_size","max_missing_days_in_window"]
    dont_match = []
    
    for prop in props:
        args_prop = dataset_args.get(prop,None)
        #Assume (maybe a bad assumption) that default args are preserved
        if args_prop is None:
            continue
        reader_prop = getattr(data_reader,prop)
        prop_match = args_prop == reader_prop

        if not prop_match:
            dont_match.append((prop,args_prop,reader_prop))
    return dont_match

def load_cached_activity_reader(name, dataset_args=None,
                                fail_if_mismatched=False,
                                activity_level="minute"):
    if not activity_level == "minute":
        raise NotImplementedError("Can only cache minute level activities")
        
    cache_path = get_cached_datareader_path(name)
    reader = pickle.load(open(cache_path, "rb" ) )
    if dataset_args:
        dont_match = validate_reader(dataset_args,reader)
        if len(dont_match) != 0:
            message = f"Mismatch between cached data reader and dataset args:{dont_match}"
            if fail_if_mismatched:
                raise ValueError(message)
            else: 
                logger.warning(message)

    elif fail_if_mismatched:
        raise(ValueError("In order to check for match with cached activity_reader must pass dataset_args"))
    return reader

def split_by_participant(df,frac):
    participants = df["participant_id"].unique()
    np.random.shuffle(participants)
    left_participants = participants[:int(frac*len(participants))]
    mask = df["participant_id"].isin(left_participants)
    left = df[mask]
    right = df[~mask]
    return left, right


def process_minute_level(minute_level_path=None, out_path =None, participant_ids=None, random_state=42):
    if minute_level_path is None:
        minute_level_df = load_raw_table("fitbit_minute_level_activity")
    else: 
        minute_level_path = dd.read_parquet(minute_level_path)
    if not participant_ids is None:
        minute_level_df = minute_level_df[minute_level_df["participant_id"].isin(participant_ids)]                                                     
        
    logger.info("Processing minute-level fitbit activity data. This will take a while...")
    # Add missing flag to heart rate
    missing_heartrate = (minute_level_df.heart_rate.isnull()) | (minute_level_df.heart_rate == 0)
    minute_level_df["missing_heartrate"] = missing_heartrate
    minute_level_df["heart_rate"] = minute_level_df["heart_rate"].fillna(0)
    # Properly encode heart rate
    minute_level_df["heart_rate"] = minute_level_df["heart_rate"].astype(int)
    

    minute_level_df['missing_steps'] = False
    minute_level_df.loc[minute_level_df.steps.isnull(),'missing_steps'] = True
    minute_level_df.steps.fillna(0,inplace = True)
    minute_level_df['missing_steps'] = minute_level_df['missing_steps'].astype(bool)
    minute_level_df['steps'] = minute_level_df['steps'].astype(np.int16)
    
    minute_level_df.sleep_classic.fillna(0,inplace = True)
    minute_level_df['sleep_classic'] = minute_level_df['sleep_classic'].astype('Int8')
    
    minute_level_df = pd.get_dummies(minute_level_df ,prefix = 'sleep_classic', columns = ['sleep_classic'],
                                    dtype = bool)
                                        
    minute_level_df.reset_index(drop = True, inplace = True)
    
    minute_level_df["date"] = minute_level_df["timestamp"].dt.date

    #Sorting will speed up dask queries later
    minute_level_df = minute_level_df.sort_values("participant_id")
    minute_level_df = minute_level_df.groupby("participant_id").apply(fill_missing_minutes)
    del minute_level_df["participant_id"]
    minute_level_df = minute_level_df.reset_index()

    minute_level_df["sleep_classic_0"] = minute_level_df["sleep_classic_0"].astype(bool)
    minute_level_df["sleep_classic_1"] = minute_level_df["sleep_classic_1"].astype(bool)
    minute_level_df["sleep_classic_2"] = minute_level_df["sleep_classic_2"].astype(bool)
    minute_level_df["sleep_classic_3"] = minute_level_df["sleep_classic_3"].astype(bool)
    # minute_level_df.to_csv("data/interim/processed_fitbit_minute_level_activity.csv")
    table = pa.Table.from_pandas(minute_level_df, preserve_index=False)

    if out_path is None:
        out_path = get_processed_dataset_path("processed_fitbit_minute_level_activity")

    pq.write_to_dataset(table, root_path=out_path,
                    partition_cols=['date'])

    paths = glob.glob(os.path.join(out_path,"*","*.parquet"))
    dd.io.parquet.create_metadata_file(paths)

def fill_missing_minutes(user_df):
    # This works because the data was pre-cleaned so that the
    # last day ends just before midnight
    min_date = user_df["timestamp"].min()
    max_date = user_df["timestamp"].max()
    new_index = pd.DatetimeIndex(pd.date_range(start=min_date,end=max_date,freq="1min"),
                                name = "timestamp")
    user_df = user_df.set_index("timestamp").reindex(new_index)
    user_df["missing_heartrate"] = user_df["missing_heartrate"].fillna(True)
    user_df["missing_steps"] = user_df["missing_steps"].fillna(True)
    user_df["steps"] = user_df["steps"].fillna(0)
    user_df["date"] = user_df.index.date
    user_df = user_df.fillna(0)
    return user_df