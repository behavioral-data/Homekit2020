import glob
import os

import click
import dask
import pandas as pd
import numpy as np
import dask.dataframe as dd

from src.models.commands import validate_dataset_args
from src.data.utils import get_dask_df, write_pandas_to_parquet, load_processed_table

@click.command()
@click.argument("out_path", type=click.Path(file_okay=False))
@click.option("--split_date",default=None)
@click.option("--eval_frac",default=None)
@click.option("--test_frac", default=0.5, help="Fraction of eval set that's reserved for testing")
@click.option("--activity_level", type=click.Choice(["day","minute"]), default="minute")
def main(out_path, split_date=None, eval_frac=None,
        test_frac = 0.5, activity_level="minute"):
    
    if activity_level == "minute":
        df = get_dask_df("processed_fitbit_minute_level_activity").compute()
        timestamp_col = "timestamp"
    else:
        df = load_processed_table("fitbit_day_level_activity")
        timestamp_col = "date"

    if split_date:
        past_date_mask = df[timestamp_col] >= pd.to_datetime(split_date)
        participants_after_date = df[past_date_mask]["participant_id"].unique()
        np.random.shuffle(participants_after_date)
        test_participants = participants_after_date[:int(test_frac*len(participants_after_date))]

        in_test_frac_mask = df["participant_id"].isin(test_participants) & past_date_mask

    elif eval_frac:
        participant_ids = df["participant_id"].unique().sample(frac=1).values
        np.random.shuffle(participant_ids)
        test_participants = participant_ids[:int(test_frac*eval_frac*len(participant_ids))]
        in_test_frac_mask  = df["participant_id"].isin(test_participants)
    
    train_eval = df[~in_test_frac_mask]
    test = df[in_test_frac_mask]


    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    train_eval_path = os.path.join(out_path,f"train_eval_{activity_level}")
    test_path = os.path.join(out_path,f"test_{activity_level}")
    
    if activity_level == "minute":
        write_pandas_to_parquet(train_eval,train_eval_path, partition_cols=["date"])
        write_pandas_to_parquet(test,test_path, partition_cols=["date"])

    else:
        test.to_csv(test_path + ".csv",index=False)
        train_eval.to_csv(train_eval_path + ".csv",index=False)
        



    

if __name__ == "__main__":
    main()