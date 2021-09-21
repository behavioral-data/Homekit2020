from asyncio.tasks import gather
from logging import Logger
from typing import Optional
import datetime as dt

from click.decorators import argument

import pyarrow.parquet as pq
import pandas as pd
import pyarrow as pa

import vaex

import logging
# from src.utils import get_logger
logger = logging.getLogger(__name__)
import click

import numpy as np
import glob
import os

def process_minute_level_pandas( minute_level_df,
                out_path =None, participant_ids=None, random_state=42):                                            

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
 
    minute_level_df = minute_level_df.groupby("participant_id").progress_apply(fill_missing_minutes)
    del minute_level_df["participant_id"]
    minute_level_df = minute_level_df.reset_index()

    minute_level_df["sleep_classic_0"] = minute_level_df["sleep_classic_0"].astype(bool)
    minute_level_df["sleep_classic_1"] = minute_level_df["sleep_classic_1"].astype(bool)
    minute_level_df["sleep_classic_2"] = minute_level_df["sleep_classic_2"].astype(bool)
    minute_level_df["sleep_classic_3"] = minute_level_df["sleep_classic_3"].astype(bool)
    # minute_level_df.to_csv("data/interim/processed_fitbit_minute_level_activity.csv")
    table = pa.Table.from_pandas(minute_level_df, preserve_index=False)


    pq.write_to_dataset(table, root_path=out_path,
                    partition_cols=['date'])



def str_col_to_list(item: str, default_len: int = 24*60,
                    sep_char: str = " ",dtype=int):
    if isinstance(item,str):
        return item.split(sep_char)
    else: 
        return [np.nan] * default_len


def explode_str_column(df, target_col: str,
                   freq: str = "min", dur: str = "1D",
                   sep_char: str = " ", date_col: str = "dt",
                   dtype: str  = "Int32", participant_col: str = "id_participant_external",
                   rename_participant_id_column: str = "participant_id",
                   rename_target_column: Optional[str] = None,
                   start_col: Optional[str] = None,
                   dur_col: Optional[str] = None,
                   return_other_cols: Optional[bool] = False) -> dict:
    df = df.to_pandas_df()
    timestamp_mapper = lambda x: get_new_index( x, target_column = target_col,
                                        start_col=start_col,
                                        dur_col=dur_col,
                                        freq=freq,
                                        dur=dur,
                                        date_col=date_col)
    df["timestamp"] = df.apply(timestamp_mapper,axis=1)
    
    _dummy_dt_start = dt.datetime.combine(dt.datetime.now().date(), dt.time.min)
    _dummy_dt_end = _dummy_dt_start + dt.timedelta(days=1)
    default_len = len(pd.date_range(start=_dummy_dt_start, end=_dummy_dt_end,
                                        freq=freq,closed="left"))

    mapper = lambda x: str_col_to_list(x, sep_char=sep_char, default_len=default_len)                                        
    df["val"] = df[target_col].apply(mapper, axis=1)
    df = df.explode(["timestamp","val"])
    df = vaex.from_pandas(df)
    # timestamps = df["timestamp"].compute()
    # pd_df = df.compute()
    # # print(timestamps.value_counts().sort_values(ascending=False))
    # # assert  timestamps.is_unique
    # assert len(pd_df.drop_duplicates(subset=["timestamp",participant_col])) == len(pd_df)
    # Need to convert to float first to handle NaN: 
    # https://stackoverflow.com/questions/39173813/pandas-convert-dtype-object-to-int
    df["val"] = df["val"].astype('float').astype(dtype)
    
    val_col_name = rename_target_column if rename_target_column else target_col
    pid_col_name = rename_participant_id_column if rename_participant_id_column else participant_col

    df = df.rename(columns={"val":val_col_name,
                            participant_col:pid_col_name})
    if return_other_cols:
        return df
    else:          
        return df[[pid_col_name,"timestamp",val_col_name]]
    

def get_new_index(item: dict, target_column=None,
                   freq: str = "min", dur: str = "1D",
                   sep_char: str = " ", date_col: str = "dt",
                   start_col: Optional[str] = None,
                   dur_col: Optional[str] = None) -> list:

    if start_col and item[start_col]:
        start = item[start_col]
        start_ts = pd.to_datetime(start).round(freq)
        end_ts = start_ts + pd.to_timedelta(item[dur_col],unit=freq)
    else:    
        start_ts = pd.to_datetime(item[date_col])
        end_ts = start_ts + pd.to_timedelta(dur)

    new_index = list(pd.date_range(start_ts,end_ts,freq=freq,closed="left").values)
    targets = item[target_column]
    if not isinstance(targets,str) and not targets:
        values = np.empty(len(new_index))
    else: 
        values = np.array(targets.split(sep_char))
        assert len(values) == len(new_index)

    return new_index

CHUNKSIZE="1GB"
PARTITION_SIZE="1GB"
def read_raw(path):
    # paths = glob.glob(os.path.join(path,"*","*.parquet"))
    # print(paths)       
    result= vaex.open(path)
    print(path)
    return result
    
@click.command()
@click.argument("sleep_in_path", type=click.Path(exists=True))
@click.argument("steps_in_path", type=click.Path(exists=True))
@click.argument("heart_rate_in_path",type=click.Path(exists=True))
@click.argument("out_path",type=click.Path())
def main(sleep_in_path: str, steps_in_path: str, 
         heart_rate_in_path: str, out_path: str) -> None:
    
    exploded_sleep = explode_str_column(read_raw(sleep_in_path),
                                    target_col = "minute_level_str",
                                    rename_target_column="sleep_classic",
                                    start_col="main_start_time",
                                    dur_col = "main_in_bed_minutes")
    
    exploded_hr = explode_str_column(read_raw(heart_rate_in_path),
                                target_col = "minute_level_str",
                                rename_target_column="heart_rate")
    
    exploded_steps = explode_str_column(read_raw(steps_in_path),
                                target_col = "minute_level_str",
                                rename_target_column="steps")
                                    
    keys = ["participant_id","timestamp"]
    # dask.compute(exploded_hr,exploded_sleep,exploded_steps)
    sleep_and_hr = exploded_sleep.merge(exploded_hr,
                                        left_on = keys,
                                        right_on = keys,
                                        how = "outer")
    
    all_streams = sleep_and_hr.merge(exploded_steps,
                                    left_on = keys,
                                    right_on = keys,
                                    how = "outer")
    # client.run(gc.collect)
    process_minute_level_pandas(minute_level_df=all_streams, out_path=out_path)

if __name__ == "__main__":
    main()