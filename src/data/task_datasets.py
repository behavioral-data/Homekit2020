from datetime import datetime

import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from distributed import Client

import dask
from dask.diagnostics import ProgressBar
import dask.dataframe as dd
dask.config.set({"distributed.comm.timeouts.connect": "60"})

from src.utils import get_logger
logger = get_logger()

from src.data.utils import get_dask_df, load_processed_table

MIN_IN_DAY = 24*60

class MinuteLevelActivityReader(object):
    def __init__(self, min_date=None,
                       max_date=None):
        
        self.dask_df = get_dask_df("processed_fitbit_minute_level_activity",
                                    min_date=min_date,
                                    max_date=max_date)

class LabResultsReader(object):
    def __init__(self,min_date=None,
                      max_date=None):
        results = load_processed_table("lab_results_with_triggerdate")

        if min_date:
            results = results[results["timestamp"].dt.date >= pd.to_datetime(min_date)]
        if max_date:
            results = results[results["timestamp"].dt.date < pd.to_datetime(max_date)]

        self.results = results

class MinuteLevelActivtyDataset(Dataset):
    def __init__(self, minute_level_activity_reader,
                      lab_results_reader,
                      day_window_size=15,
                      max_missing_days_in_window=5,
                      min_windows=1,
                      pos_participants_only=True):
        
        with Client(n_workers=32,threads_per_worker=1) as client:
            dask_df = minute_level_activity_reader.dask_df
            self.lab_results = lab_results_reader.results.set_index("participant_id")
           
            
            if pos_participants_only:
                self.lab_results = self.lab_results[self.lab_results["result"] == 'Detected']
            
            self.participant_ids = self.lab_results.index.unique().values
            dask_df = dask_df[dask_df["participant_id"].isin(self.participant_ids)]                

            self.day_window_size = day_window_size
            self.max_missing_days_in_window = max_missing_days_in_window
            self.min_windows = min_windows
            self.pos_participants_only = pos_participants_only

            valid_participant_dates = dask_df.groupby("participant_id").apply(self.get_valid_dates, meta=("dates",object)).compute()
            
            self.minute_data = dask_df
            with ProgressBar():
                logger.info("Using Dask to pre-process data:")
                valid_particpant_dates, self.minute_data = dask.compute(valid_participant_dates,self.minute_data)
            self.minute_data = self.minute_data.set_index(["participant_id","timestamp"])
            self.valid_participant_dates = list(valid_participant_dates.apply(pd.Series).stack().droplevel(-1).items())
          
    def get_user_data_in_time_range(self,participant_id,start,end):
        return self.minute_data.loc[participant_id].loc[start:end]
    
    def look_for_test_result(self,participant_id,date):
        try:
            participant_results = self.lab_results.loc[participant_id]
        except KeyError:
            return None

        result = participant_results["trigger_datetime"].date == date
        if result:
            return result["result"]
        else:
            return None
        

    def get_valid_dates(self, partition):
        dates_with_data = pd.DatetimeIndex(partition[~partition["missing_heartrate"]]["timestamp"].dt.date.unique())
        min_days_with_data = self.day_window_size - self.max_missing_days_in_window
       
        if len(dates_with_data) < min_days_with_data:
            return 
        min_date = dates_with_data.min()
        max_date = dates_with_data.max()

        all_possible_start_dates = pd.date_range(min_date,max_date-pd.Timedelta(days=self.day_window_size))
        all_possible_end_dates = all_possible_start_dates + pd.Timedelta(days=self.day_window_size)

        min_days_with_data = self.day_window_size - self.max_missing_days_in_window

        mask = []
        for a,b in zip(all_possible_start_dates, all_possible_end_dates):
            has_enough_data = len(dates_with_data[(dates_with_data >= a) & (dates_with_data < b) ] )> min_days_with_data
            mask.append(has_enough_data)

        return all_possible_end_dates[mask].rename("dates")


    def __len__(self):
        return len(self.valid_participant_dates)
    
    def __getitem__(self,index):
        # Could cache this later
        participant_id, end_date = self.valid_participant_dates[index]
        start_date = end_date - pd.Timedelta(self.day_window_size, unit = "days")
        minute_data = self.get_user_data_in_time_range(participant_id,start_date,end_date)
        
        result = self.look_for_test_result(participant_id,end_date)
        if result and result == "Detected":
            label = 1
        else:
            label = 0
        
        return {
            "label": label,
            "data" : minute_data
        }
    
if __name__ == "__main__":
    minute_level_reader = MinuteLevelActivityReader()
    lab_results_reader = LabResultsReader()
    dataset =  MinuteLevelActivtyDataset(minute_level_reader, lab_results_reader)
