from typing import Optional
import datetime as dt
import gc
import glob
import os

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.utils import  process_minute_level_pandas
from src.utils import get_logger
logger = get_logger(__name__)

import click
from tqdm import tqdm

import os

import numpy as np
import pandas as pd

import time

def explode_str_column(df: pd.DataFrame, target_col: str,
                   freq: str = "min", dur: str = "1D",
                   sep_char: str = " ", date_col: str = "dt",
                   dtype: str  = "Int32", participant_col: str = "id_participant_external",
                   rename_participant_id_column: str = "participant_id",
                   rename_target_column: Optional[str] = None,
                   start_col: Optional[str] = None,
                   dur_col: Optional[str] = None,
                   clip_max: int = 200,
                   single_val=False) -> dict:

    # tqdm.pandas(desc="Getting new indices...")
    # logger.info("Getting new indices....")
    val_col_name = rename_target_column if rename_target_column else target_col
    pid_col_name = rename_participant_id_column if rename_participant_id_column else participant_col

    if df.empty:
        return pd.DataFrame(columns=["timestamp",val_col_name],index=pd.DatetimeIndex([])).set_index("timestamp")

    df["timestamp"] = df.apply(get_new_index,target_column = target_col,
                                    start_col=start_col,
                                    dur_col=dur_col,
                                    freq=freq,
                                    dur=dur,
                                    axis=1,
                                    date_col=date_col)


    # mapper = lambda x: str_col_to_list(x, sep_char=sep_char, default_len=default_len)                                        
    if not single_val:
        df["val"] = df[target_col].str.split(sep_char)
    else:
        df["val"] = df.apply(lambda x: [x[target_col]] * len(x["timestamp"]), axis=1)
    # logger.info("Exploding column...")
    df = df[["timestamp","val"]].explode(["timestamp","val"])
    df["val"] = pd.to_numeric(df["val"],downcast="unsigned").clip(upper=clip_max)
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset="timestamp",keep="last")
    df = df.rename(columns={"val":val_col_name})
    # logger.info("Setting index...")
    df = df.set_index("timestamp").sort_index()
    return df
    
def get_new_index(item: dict, target_column: str,
                   freq: str = "min", dur: str = "1D",
                   sep_char: str = " ", date_col: str = "dt",
                   start_col: Optional[str] = None,
                   dur_col: Optional[str] = None) -> list:

    if start_col:
        start = item[start_col]
        start_ts = pd.to_datetime(start).round(freq)
        end_ts = start_ts + pd.to_timedelta(item[dur_col],unit=freq)
    else:    
        start_ts = pd.to_datetime(item[date_col])
        end_ts = start_ts + pd.to_timedelta(dur)

    new_index = pd.date_range(start_ts,end_ts,freq=freq,closed="left").values
    return new_index

CHUNKSIZE="1GB"
PARTITION_SIZE="1GB"

def read_raw_pandas(path,set_dtypes=None):
    logger.info("Reading...")
    df = pd.read_parquet(path,engine='pyarrow')
    df["id_participant_external"] = df["id_participant_external"].astype("category")
    if set_dtypes:
        for k,v in set_dtypes.items():
            df[k] = df[k].astype(v)
            
    return df.dropna().set_index("id_participant_external")

def safe_loc(df,ind):
    try:
        return df.loc[[ind]]
    except KeyError:
        return pd.DataFrame(columns=df.columns)

COLUMNS = ["date",
           "timestamp",
           "heart_rate",
           "steps",
           "missing_heart_rate",
           "missing_steps",
           "sleep_classic_0",
           "sleep_classic_1",
           "sleep_classic_2",
           "sleep_classic_3"]

@click.command()
@click.argument("sleep_in_path", type=click.Path(exists=True))
@click.argument("steps_in_path", type=click.Path(exists=True))
@click.argument("heart_rate_in_path",type=click.Path(exists=True))
@click.argument("out_path",type=click.Path())
def main(sleep_in_path: str, steps_in_path: str, 
         heart_rate_in_path: str, out_path: str) -> None:
    
    start = time.time()
    logger.info("Loading sleep...")
    sleep = read_raw_pandas(sleep_in_path)

    logger.info("Loading heart rate...") 
    hr = read_raw_pandas(heart_rate_in_path)

    logger.info("Loading steps...") 
    steps = read_raw_pandas(steps_in_path)

    users_with_steps = steps.index.unique()

    logger.info("Processing users...")               
    all_results = []

    for user in tqdm(users_with_steps.values):
        exploded_sleep = explode_str_column(safe_loc(sleep,user),
                                    target_col = "minute_level_str",
                                    rename_target_column="sleep_classic",
                                    start_col="main_start_time",
                                    dur_col = "main_in_bed_minutes",
                                    dtype=pd.Int8Dtype())
        exploded_hr =  explode_str_column(safe_loc(hr,user),
                                          target_col = "minute_level_str",
                                          rename_target_column="heart_rate",
                                          dtype=pd.Int8Dtype())
        exploded_steps = explode_str_column(safe_loc(steps,user),
                                            target_col="minute_level_str",
                                            rename_target_column="steps",
                                            dtype=pd.Int8Dtype())
        steps_and_hr = exploded_steps.join(exploded_hr,how = "left") 
        merged = steps_and_hr.join(exploded_sleep,how="left")                        

        
        processed = process_minute_level_pandas(minute_level_df = merged)

        # Keep datatypes in check
        processed["heart_rate"] = processed["heart_rate"].astype(pd.Int16Dtype())
        processed["participant_id"] = user
        all_results.append(processed)

    all_results = pd.concat(all_results)
    all_results["sleep_classic_0"] = all_results["sleep_classic_0"].fillna(False)
    all_results["sleep_classic_1"] = all_results["sleep_classic_1"].fillna(False)
    all_results["sleep_classic_2"] = all_results["sleep_classic_2"].fillna(False)
    all_results["sleep_classic_3"] = all_results["sleep_classic_3"].fillna(False)

    all_results.to_parquet(path = out_path, partition_cols=["date"], engine="fastparquet")
    end = time.time()
    print("Time elapsed",end-start)
if __name__ == "__main__":
    main()