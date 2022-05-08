import glob
import os

import click
import pandas as pd
import numpy as np

from src.data.utils import  write_pandas_to_parquet, load_processed_table, read_parquet_to_pandas

@click.command()
@click.argument("out_path", type=click.Path(file_okay=False))
@click.argument("in_path", type=click.Path())
@click.option("--split_date",default=None)
@click.option("--end_date",default=None)
@click.option("--eval_frac",default=None)
@click.option("--test_frac", default=0.5, help="Fraction of eval set that's reserved for testing")
@click.option("--activity_level", type=click.Choice(["day","minute"]), default="minute")
@click.option("--separate_train_and_eval", is_flag=True)
def main(out_path, in_path, split_date=None, end_date=None, eval_frac=None,
        test_frac = 0.5, activity_level="minute",
        separate_train_and_eval=False):
    
    if activity_level == "minute":
        df = read_parquet_to_pandas(in_path)
        timestamp_col = "timestamp"
    else:
        df = load_processed_table("fitbit_day_level_activity")
        timestamp_col = "date"

    if end_date:
        df = df[df[timestamp_col] < pd.to_datetime(end_date)]

    if split_date:
        past_date_mask = df[timestamp_col] >= pd.to_datetime(split_date)
        participants_after_date = df[past_date_mask]["participant_id"].unique()
        np.random.shuffle(participants_after_date)
        test_participants = participants_after_date[:int(test_frac*len(participants_after_date))]

        in_test_frac_mask = df["participant_id"].isin(test_participants) & past_date_mask

    elif eval_frac:
        participant_ids = df["participant_id"].unique().values
        np.random.shuffle(participant_ids)
        test_index = int(test_frac*eval_frac*len(participant_ids))
        eval_index = int(eval_frac*len(participant_ids))
        test_participants = participant_ids[:test_index]
        eval_participants = participant_ids[test_index:eval_index]
        in_test_frac_mask  = df["participant_id"].isin(test_participants)
    
    train_eval = df[~in_test_frac_mask]
    test = df[in_test_frac_mask]
    test_path = os.path.join(out_path,f"test_{activity_level}")
    
    to_write_dfs = [test]
    to_write_paths =[test_path]
    
    if separate_train_and_eval:
        if split_date:
            train = train_eval[train_eval[timestamp_col] < pd.to_datetime(split_date)]
            eval = train_eval[train_eval[timestamp_col] >= pd.to_datetime(split_date)]
        elif eval_frac:
            eval_mask = train_eval["participant_id"].isin(eval_participants)
            train = train_eval[~eval_mask]
            eval = train_eval[eval_mask]
        train_path = os.path.join(out_path,f"train_{activity_level}")
        eval_path = os.path.join(out_path,f"eval_{activity_level}")

        to_write_dfs = to_write_dfs + [train,eval]
        to_write_paths = to_write_paths + [train_path,eval_path]
    else:
        train_eval_path = os.path.join(out_path,f"train_eval_{activity_level}")
        
        to_write_dfs += [train_eval]
        to_write_paths += [train_eval_path]


    if not os.path.exists(out_path):
        os.mkdir(out_path)

    df["participant_id"] = df["participant_id"].astype("string")
    if activity_level == "minute":
        for df, path in zip(to_write_dfs,to_write_paths):
            write_pandas_to_parquet(df,path, partition_cols=["date"],overwrite=True,
                                    engine="fastparquet")

    else:
        for df, path in zip(to_write_dfs,to_write_paths):
            df.to_csv(path + ".csv",index=False)
        
if __name__ == "__main__":
    main()