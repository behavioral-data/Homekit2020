import os
import glob

import click
import dask
import pandas as pd
import numpy as np
import dask.dataframe as dd

from src.data.utils import get_dask_df, write_pandas_to_parquet

@click.command()
@click.argument("out_path", type=click.Path(file_okay=False))
@click.option("--split_date",default=None)
@click.option("--test_frac", default=None, help="Fraction of eval set that's reserved for testing")
def main(out_path, split_date=None, 
        test_frac = None, activity_level="minute"):
    if not split_date and not test_frac:
        raise ValueError("Must pass either 'split_date' or 'test_frac'")
        
    dask_df = get_dask_df("processed_fitbit_minute_level_activity").compute()
    if split_date:
        past_date_mask = dask_df["timestamp"] >= pd.to_datetime(split_date)
        participants_after_date = dask_df[past_date_mask]["participant_id"].unique()
        np.random.shuffle(participants_after_date)
        test_participants = participants_after_date[:int(test_frac*len(participants_after_date))]

        in_test_frac_mask = dask_df["participant_id"].isin(test_participants) & past_date_mask

    elif test_frac:
        participant_ids = dask_df["participant_id"].unique()
        np.random.shuffle(participant_ids)
        test_participants = participant_ids[:int(test_frac*len(participant_ids))]
        in_test_frac_mask  = dask_df["participant_id"].isin(test_participants)
    
    
    train_eval = dask_df[~in_test_frac_mask]
    test = dask_df[in_test_frac_mask]


    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    train_eval_path = os.path.join(out_path,f"train_eval_{activity_level}")
    write_pandas_to_parquet(train_eval,train_eval_path, partition_cols=["date"])

    test_path = os.path.join(out_path,f"test_{activity_level}")
    write_pandas_to_parquet(test,test_path, partition_cols=["date"])



    

if __name__ == "__main__":
    main()