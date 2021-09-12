from asyncio.tasks import gather
from typing import Optional
import datetime as dt
import gc

from src.data.utils import get_dask_df, process_minute_level, process_minute_level_pandas

import click

import dask.dataframe as dd
import dask
from dask.distributed import Client, LocalCluster
from tqdm import tqdm

import numpy as np
import pandas as pd


def str_col_to_list(item: str, default_len: int = 24*60,
                    sep_char: str = " ",dtype=int):
    if isinstance(item,str):
        return item.split(sep_char)
    else: 
        return [np.nan] * default_len


def explode_str_column(df: dd, target_col: str,
                   freq: str = "min", dur: str = "1D",
                   sep_char: str = " ", date_col: str = "dt",
                   dtype: str  = "Int32", participant_col: str = "id_participant_external",
                   rename_participant_id_column: str = "participant_id",
                   rename_target_column: Optional[str] = None,
                   start_col: Optional[str] = None,
                   dur_col: Optional[str] = None,
                   return_other_cols: Optional[bool] = False) -> dict:

    df["timestamp"] = df.apply(get_new_index,target_column = target_col,
                                    start_col=start_col,
                                    dur_col=dur_col,
                                    freq=freq,
                                    dur=dur,
                                    axis=1,
                                    date_col=date_col)
    
    _dummy_dt_start = dt.datetime.combine(dt.datetime.now().date(), dt.time.min)
    _dummy_dt_end = _dummy_dt_start + dt.timedelta(days=1)
    default_len = len(pd.date_range(start=_dummy_dt_start, end=_dummy_dt_end,
                                        freq=freq,closed="left"))

    mapper = lambda x: str_col_to_list(x, sep_char=sep_char, default_len=default_len)                                        
    df["val"] = df[target_col].map(mapper)
    df = df.explode(["timestamp","val"])
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
    

def get_new_index(item: dict, target_column: str,
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
        df = dd.read_parquet(path,
                            # aggregate_files="id_participant_external", 
                            engine='pyarrow-legacy', 
                            index="id_participant_external",
                            chunksize=PARTITION_SIZE)
        return df 
@click.command()
@click.argument("sleep_in_path", type=click.Path(exists=True))
@click.argument("steps_in_path", type=click.Path(exists=True))
@click.argument("heart_rate_in_path",type=click.Path(exists=True))
@click.argument("out_path",type=click.Path())
def main(sleep_in_path: str, steps_in_path: str, 
         heart_rate_in_path: str, out_path: str) -> None:
    

    with dask.config.set({  "distributed.memory.target": 0.95,  # target fraction to stay below
                            "distributed.memory.spill": False, # fraction at which we spill to disk
                            "distributed.memory.pause": False,  # fraction at which we pause worker threads
                            "distributed.memory.terminate": False}):
        print(dask.config.get("distributed.memory.spill"))
        # cluster = LocalCluster(n_workers=8, 
        #                         threads_per_worker=8,
        #                         memory_limit='256GB',
        #                         local_directory="/projects/bdata/mikeam_scratch")
        cluster = LocalCluster(local_directory="/tmp")
        
        client = Client(cluster)
        # def enable_fault_handler():
        #     import faulthandler
        #     faulthandler.enable()
        #     print('enabled fault handler')

        # # run it locally
        # enable_fault_handler()

        # # run it on all workers
        # client.run(enable_fault_handler)

        # # run it on the scheduler (might fail, but no problem)
        # client.run_on_scheduler(enable_fault_handler)

        # sleep = read_raw(sleep_in_path).compute()
        # hr = read_raw(heart_rate_in_path).compute()
        # steps = read_raw(steps_in_path).compute()
        # # print(f"Raw divisions are: {steps.known_divisions}")
        # keys = ["id_participant_external","dt"]
        # sleep_and_hr = sleep.merge(hr, on=keys, how = "outer",suffixes = ["sleep","hr"])
        # all_streams = sleep_and_hr.merge(steps, on=keys, how = "outer")
        
        # print(f"Merged divisions are: {all_streams.known_divisions}")
        # dd.to_parquet(all_streams, out_path, partition_on=["dt"], write_metadata_file=False, overwrite=True)
    # sleep_and_hr = exploded_sleep.merge(exploded_hr,
    #                                     left_on = keys,
    #                                     right_on = keys,
    #                                     how = "outer")

    #                                                     gather_statistics=True,aggregate_files=True,
    #                                                     engine='pyarrow-legacy'),#.repartition(partition_size=PARTITION_SIZE),
    #                             target_col = "minute_level_str",
    #                             rename_target_column="steps")
    exploded_sleep = explode_str_column(pd.read_parquet(sleep_in_path,
                                                        engine='pyarrow'),#.repartition(partition_size=PARTITION_SIZE), 
                                    target_col = "minute_level_str",
                                    rename_target_column="sleep_classic",
                                    start_col="main_start_time",
                                    dur_col = "main_in_bed_minutes")
    
    exploded_hr = explode_str_column(pd.read_parquet(heart_rate_in_path,
                                                    engine='pyarrow'),
                                target_col = "minute_level_str",
                                rename_target_column="heart_rate")
    
    exploded_steps = explode_str_column(pd.read_parquet(steps_in_path,
                                                        engine='pyarrow'),
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