# -*- coding: utf-8 -*-
"""
README
This is an old script from when this library relied on "loader"
functions (e.g. "load_processed_table") to load data. All datasets besides
the audere dataset (/projects/bdata/datasets/gatesfoundation/audere) should 
use make_dataset.py
"""
from src.data.make_dataset import process_minute_level
import click
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import dask.dataframe as dd

import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import glob

from src.utils import get_logger
import os
logger = get_logger(__name__)

from src.data.utils import (load_raw_table, load_processed_table, get_processed_dataset_path,
                            process_minute_level)
import src.data.constants as cons



def process_day_level(participant_ids=None, random_state=42):
    day_level_df = load_raw_table("fitbit_day_level_activity")
    if not participant_ids is None:
        day_level_df = day_level_df[day_level_df["participant_id"].isin(participant_ids)]  

    day_level_df = day_level_df.drop(columns=["main_start_time"])
    day_level_df["missing_hr"] = day_level_df["resting_heart_rate"].isnull().astype(int)
    day_level_df.fillna(value = {'resting_heart_rate': 0}, inplace = True)

    day_level_df["missing_sleep"] = day_level_df["total_asleep_minutes"].isnull().astype(int)
    day_level_df.fillna(value = {k:0 for k in cons.DAILY_SLEEP_FEATURES}, inplace=True)    

    day_level_df["missing_steps"] = day_level_df["activityCalories"].isnull().astype(int)
    day_level_df.fillna(value = {k:0 for k in cons.DAILY_STEP_FEATURES}, inplace=True) 
    
    day_level_df = day_level_df.groupby("participant_id").apply(fill_missing_days)
    del day_level_df["participant_id"]
    day_level_df = day_level_df.reset_index()

    processed_path = get_processed_dataset_path("fitbit_day_level_activity")
    day_level_df.to_csv(processed_path)


def fill_missing_days(user_df):
    min_date = user_df["date"].min()
    max_date = user_df["date"].max()
    user_df["missing_day"] = 0

    new_index = pd.DatetimeIndex(pd.date_range(start=min_date,end=max_date,freq="1D"),
                                name = "date")
                                
    user_df = user_df.set_index("date").reindex(new_index)
    user_df["missing_steps"] = user_df["missing_steps"].fillna(1)
    user_df["missing_hr"] = user_df["missing_steps"].fillna(1)
    user_df["missing_sleep"] = user_df["missing_sleep"].fillna(1)
    user_df["missing_day"] = user_df["missing_sleep"].fillna(1)

    user_df = user_df.fillna(0)
    return user_df


def process_surveys(return_result=False):
    # Ported from notebooks/melih_notebooks/EDA_survey.ipynb
    # Produces lab_results_with_trigger, baseline_screener_survey, daily_surveys_onehot

    df_baseline = load_raw_table("baseline")
    df_fup_a = load_raw_table("follow_up_a")
    df_fup_b = load_raw_table("follow_up_b")
    df_screener = load_raw_table("screener")

    df_trigerred_fup_a = df_fup_a.loc[
             (df_fup_a.symptom_severity__cough_q != '0') & (df_fup_a.symptom_severity__cough_q != 'skipped') &  
             (((df_fup_a.symptom_severity__aches_q != '0') & (df_fup_a.symptom_severity__aches_q != 'skipped')) 
             | ((df_fup_a.symptom_severity__chills_q != '0') & (df_fup_a.symptom_severity__chills_q != 'skipped')) 
             | ((df_fup_a.symptom_severity__fever_q != '0') & (df_fup_a.symptom_severity__fever_q != 'skipped')) 
             | ((df_fup_a.symptom_severity__sweats_q != '0') & (df_fup_a.symptom_severity__sweats_q != 'skipped')))
            ]
    
    #Selecting when kits were received for each participant
    df_lab_updates = load_processed_table("lab_updates")
    df_lab_updates_received = df_lab_updates.loc[(~df_lab_updates['received_datetime'].isnull())]
    df_lab_updates_received = df_lab_updates_received[['participant_id','received_datetime']]

    df_lab_results = load_processed_table("lab_results")
    
    # Matching the kit received date with lab results
    df_lab_results = pd.merge(df_lab_results, df_lab_updates_received,  how = 'left', on = 'participant_id')

    df_lab_results_w_triggerdate = pd.merge(df_lab_results, df_trigerred_fup_a.sort_values('timestamp').drop_duplicates('participant_id')[['participant_id','timestamp','first_report_yn']]
         , how = 'left', on = 'participant_id')
    df_lab_results_w_triggerdate.rename(columns = {"timestamp":"trigger_datetime"}, inplace = True)
    
    lab_updates_with_trigger_path = get_processed_dataset_path("lab_results_with_triggerdate")
    df_lab_results_w_triggerdate.to_csv(lab_updates_with_trigger_path,index=False)
    
    baseline_screener_survey_path = get_processed_dataset_path("baseline_screener_survey")
    pd.merge(df_screener[['participant_id','timestamp', 'sex', 'ethnicity', 'race__0', 'race__1',
       'race__2', 'race__3', 'race__4', 'race__5', 'postal_code', 'age']], df_baseline
         , how = 'left', on = 'participant_id' ).to_csv(baseline_screener_survey_path)

    #One hot encode survey data:
    df_daily_survey = pd.concat([df_fup_a,df_fup_b])
    no_one_hot = ['occurrence', 'timestamp', 'participant_id', 'have_flu', 'recovered_yn',
        'recovery_datetime','first_report_yn', 'first_sx_datetime', 'body_temp_f']
    one_hot_encode = list(set(df_daily_survey.columns.values) - set(no_one_hot))
    for col in one_hot_encode:
            df_daily_survey = pd.get_dummies(df_daily_survey ,prefix = col, columns = [col])
    df_daily_survey['have_flu'] = df_daily_survey['have_flu'].astype('uint8')

    one_hot_path = get_processed_dataset_path("daily_surveys_onehot")
    df_daily_survey.to_csv(one_hot_path,index=False)

    if return_result:
        return df_daily_survey

def lab_updates(updates_path=None,return_result=False):
    if not updates_path:
        df_updates = load_raw_table("mtl_lab_order_updates")
    else:
        df_updates = load_raw_table(updates_path)

    # From notebooks/melih_notebooks/EDA_mtl_lab.ipynb
    df_updates['shipped_datetime'] = pd.to_datetime(df_updates.shipped_datetime)

    df_updates['received_datetime'] = pd.to_datetime(df_updates.received_datetime)
    df_updates['report_sent_datetime'] = pd.to_datetime(df_updates.report_sent_datetime)

    lab_updates_path = get_processed_dataset_path("lab_updates")
    df_updates.to_csv(lab_updates_path,index=False)
    
    if return_result:
        return df_updates

def lab_results(return_result=False):
    df_results = load_raw_table("mtl_lab_order_results")

    # From notebooks/melih_notebooks/EDA_mtl_lab.ipynb
    df_results['report_sent_datetime'] = pd.to_datetime(df_results.report_sent_datetime)
    df_results['assay_datetime'] = pd.to_datetime(df_results.assay_datetime)

    lab_results_path = get_processed_dataset_path("lab_results")
    df_results.to_csv(lab_results_path,index=False)

    if return_result:
        return df_results

# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Lab updates:
    lab_updates()
    lab_results()

    # Surveys
    process_surveys()

    # Minute Level Activity
    process_minute_level()
    process_day_level()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    # pylint: disable=no-value-for-parameter
    main()
