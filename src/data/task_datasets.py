from datetime import datetime

import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from distributed import Client
import dask
from dask.diagnostics import ProgressBar
import dask.dataframe as dd
dask.config.set({"distributed.comm.timeouts.connect": "60"})

from sklearn.preprocessing import MinMaxScaler

from src.utils import get_logger
logger = get_logger()

from src.data.utils import get_dask_df, load_processed_table

MIN_IN_DAY = 24*60

class MinuteLevelActivityReader(object):
    def __init__(self, min_date=None,
                       split_date=None,
                       max_date=None,
                       participant_ids=None,
                       scaler=MinMaxScaler,
                       day_window_size=15,
                       max_missing_days_in_window=5,
                       min_windows=1):
        
        self.scaler = scaler
        self.day_window_size = day_window_size
        self.max_missing_days_in_window = max_missing_days_in_window
        self.min_windows = min_windows
        
        dask_df = get_dask_df("processed_fitbit_minute_level_activity",
                                    min_date=min_date,
                                    max_date=max_date)
    
        if not participant_ids is None:
            dask_df = dask_df[dask_df["participant_id"].isin(participant_ids)] 

        # pylint: disable=unused-variable
        with Client(n_workers=32,threads_per_worker=1) as client:         

            self.day_window_size = day_window_size
            self.max_missing_days_in_window = max_missing_days_in_window
            self.min_windows = min_windows

            valid_participant_dates = dask_df.groupby("participant_id").apply(self.get_valid_dates, meta=("dates",object))
            
            self.minute_data = dask_df
            with ProgressBar():
                logger.info("Using Dask to pre-process data:")
                valid_participant_dates, self.minute_data = dask.compute(valid_participant_dates,self.minute_data)
            
            self.minute_data = self.minute_data.set_index(["participant_id","timestamp"]).drop(columns=["date"])
            self.participant_dates = list(valid_participant_dates.apply(pd.Series).stack().droplevel(-1).items())
        
        if self.scaler:
            scale_model = self.scaler()
            self.minute_data[self.minute_data.columns] = scale_model.fit_transform(self.minute_data)

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

    def split_participant_dates(self,date):
        """Split participant dates to be before and after a given date"""
        before_mask = self.participant_dates.index(level=-1) <= pd.to_datetime(date)
        before = self.participant_dates[before_mask]
        after = self.participant_dates[~before_mask]
        return before, after

class LabResultsReader(object):
    def __init__(self,min_date=None,
                      max_date=None,
                      pos_only=True):
        results = load_processed_table("lab_results_with_triggerdate").set_index("participant_id")

        if min_date:
            results = results[results["timestamp"].dt.date >= pd.to_datetime(min_date)]
        if max_date:
            results = results[results["timestamp"].dt.date < pd.to_datetime(max_date)]

        if pos_only:
            results = results[results["result"] == 'Detected']
        
        self.results = results
        self.participant_ids = self.results.index.unique().values

class MinuteLevelActivtyDataset(Dataset):
    def __init__(self, minute_level_activity_reader,
                       lab_results_reader,
                       participant_dates,
                       day_window_size=15,
                       max_missing_days_in_window=5,
                       min_windows=1,
                       scaler = MinMaxScaler):
        
        self.minute_level_activity_reader = minute_level_activity_reader
        self.minute_data = self.minute_level_activity_reader.minute_data
        self.day_window_size = self.minute_level_activity_reader.day_window_size 

        self.lab_results_reader = lab_results_reader
        self.participant_dates = participant_dates
        

    def get_user_data_in_time_range(self,participant_id,start,end):
        return self.minute_data.loc[participant_id].loc[start:end]
    
    def look_for_test_result(self,participant_id,date):
        try:
            participant_results = self.lab_results_reader.results.loc[participant_id]
        except KeyError:
            return None

        result = participant_results["trigger_datetime"].date == date
        if result:
            return result["result"]
        else:
            return None


    def __len__(self):
        return len(self.participant_dates)
    
    def __getitem__(self,index):
        # Could cache this later
        participant_id, end_date = self.participant_dates[index]
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
    
    lab_results_reader = LabResultsReader()
    participant_ids = lab_results_reader.participant_ids
    minute_level_reader = MinuteLevelActivityReader(participant_ids=participant_ids)

    dataset = MinuteLevelActivtyDataset(minute_level_reader, lab_results_reader,
                                        participant_dates = minute_level_reader.participant_dates)

