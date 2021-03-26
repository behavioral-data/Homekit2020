from datetime import datetime
import multiprocessing
from functools import lru_cache, reduce
from operator import and_ as bit_and
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from distributed import Client
import dask
from dask.diagnostics import ProgressBar
import dask.dataframe as dd
dask.config.set({"distributed.comm.timeouts.connect": "60"})

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from src.utils import get_logger
logger = get_logger()

from src.data.utils import get_dask_df, load_processed_table

from tqdm import tqdm

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
        
        n_cores = multiprocessing.cpu_count()
        pbar = ProgressBar()
        pbar.register()
        
        #pylint:disable=unused-variable 
        with Client(n_workers=min(n_cores,16), threads_per_worker=1) as client:

            dask_df = get_dask_df("processed_fitbit_minute_level_activity")
        
            if not participant_ids is None:
                dask_df = dask_df[dask_df["participant_id"].isin(participant_ids)] 

            date_filters = []
            
            if min_date: 
                min_datetime = pd.to_datetime(min_date)
                date_filters.append(dask_df["timestamp"] >= min_datetime)
            
            if max_date:
                max_datetime = pd.to_datetime(max_date)
                date_filters.append(dask_df["timestamp"] < max_datetime)
            filters = reduce(bit_and,date_filters)

            if not len(filters) == 0:
                dask_df = dask_df[filters]
            
            self.day_window_size = day_window_size
            self.max_missing_days_in_window = max_missing_days_in_window
            self.min_windows = min_windows
            

            valid_participant_dates = dask_df.groupby("participant_id").apply(self.get_valid_dates, meta=("dates",object))
            
            self.minute_data = dask_df
        
            logger.info("Using Dask to pre-process data:")
            valid_participant_dates, self.minute_data = dask.compute(valid_participant_dates,self.minute_data)
            
            self.minute_data = self.minute_data.set_index(["participant_id","timestamp"]).drop(columns=["date"])
            self.participant_dates = list(valid_participant_dates.dropna().apply(pd.Series).stack().droplevel(-1).items())
        
        if self.scaler:
            scale_model = self.scaler()
            self.minute_data[self.minute_data.columns] = scale_model.fit_transform(self.minute_data)
        self.minute_data = self.minute_data.sort_index()

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
            has_enough_data = len(dates_with_data[(dates_with_data >= a) & (dates_with_data < b) ] )>= min_days_with_data
            mask.append(has_enough_data)

        return all_possible_end_dates[mask].rename("dates")

    def split_participant_dates(self,date=None,eval_frac=None):
        """If random, split a fraction equal to random for eval,
            else, split participant dates to be before and after a given date"""

        if eval_frac:
            train,eval = train_test_split(self.participant_dates,test_size=eval_frac)
            return train,eval
        elif date:
            before = [x for x in self.participant_dates if x[1] <= pd.to_datetime(date) ]
            after = [x for x in self.participant_dates if x[1] > pd.to_datetime(date) ]
            return before, after
        else:
            raise ValueError("If splitting, must either provide a date or fraction")

class LabResultsReader(object):
    def __init__(self,min_date=None,
                      max_date=None,
                      pos_only=False):
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
                       scaler = MinMaxScaler,
                       time_encoding=None,
                       return_dict=False,
                       return_global_attention_mask = False,
                       add_cls=False,
                       shuffle=False):
        
        self.minute_level_activity_reader = minute_level_activity_reader
        self.minute_data = self.minute_level_activity_reader.minute_data
        self.day_window_size = self.minute_level_activity_reader.day_window_size 

        self.lab_results_reader = lab_results_reader
        
        self.participant_dates = participant_dates
        if shuffle:
            random.shuffle(self.participant_dates)
        
        self.time_encoding = time_encoding
        self.return_dict = return_dict
        self.return_global_attention_mask = return_global_attention_mask
        
        self.size = (24*60*day_window_size+1+int(add_cls), 2*bool(time_encoding) + 8)
        self.add_cls = add_cls

        if self.add_cls:
            self.cls_init = np.random.randn(1,self.size[-1]).astype(np.float32)   
        
    def get_user_data_in_time_range(self,participant_id,start,end):
        data = self.minute_data.loc[participant_id].loc[start:end]
        return data
    
    def get_label(self,participant_id,start_date,end_date):
        try:
            participant_results = self.lab_results_reader.results.loc[participant_id]
        except KeyError:
            return None
        
        # Indexing returns a series when there's only one result
        if type(participant_results) == pd.Series:
            participant_results = participant_results.to_frame().T
        
        on_date = participant_results["trigger_datetime"] == end_date.date()
        is_pos = participant_results["result"] == "Detected"
        return any(on_date & is_pos)

    def __len__(self):
        return len(self.participant_dates)
    
    @lru_cache(maxsize=None)
    def __getitem__(self,index):
        # Could cache this later
        participant_id, end_date = self.participant_dates[index]
        start_date = end_date - pd.Timedelta(self.day_window_size, unit = "days")
        minute_data = self.get_user_data_in_time_range(participant_id,start_date,end_date)
        
        result = self.get_label(participant_id,start_date,end_date)
        if result:
            label = 1
        else:
            label = 0
        
        if self.time_encoding == "sincos":
            minute_data["sin_time"]  = sin_time(minute_data.index)
            minute_data["cos_time"]  = cos_time(minute_data.index)
        
        
        if self.return_dict:
            
            item = {}
            embeds = minute_data.values.astype(np.float32)        
            
            if self.add_cls:
                embeds = np.concatenate([self.cls_init,embeds],axis=0)
            
            item["inputs_embeds"] = embeds
            item["label"] = label

            if self.return_global_attention_mask:
                mask = np.zeros(embeds.shape[0])
                mask[0] = 1
                item["global_attention_mask"] = mask
            
            return item

        return minute_data.values, label
    
    def to_stacked_numpy(self):
        
        if len(self)==0:
            return np.array([]), np.array([])
        X = []
        y = []

        for el_x, el_y in tqdm(self, desc = "Converting to np Array"): 
            X.append(el_x)
            y.append(el_y)
        
        return np.stack(X), np.stack(y)


class EarlyDetectionDataset(MinuteLevelActivtyDataset):
    def __init__(self,*args,**kwargs):
        super(EarlyDetectionDataset, self).__init__(*args, **kwargs)

class PredictTriggerDataset(MinuteLevelActivtyDataset):
    def __init__(self,*args,**kwargs):
        super(PredictTriggerDataset, self).__init__(*args, **kwargs)
    
    def get_label(self, participant_id, start_date, end_date):
        try:
            participant_results = self.lab_results_reader.results.loc[participant_id]
        except KeyError:
            return None
        
        # Indexing returns a series when there's only one result
        if type(participant_results) == pd.Series:
            participant_results = participant_results.to_frame().T
        
        on_date = participant_results["trigger_datetime"].dt.date == end_date.date()
        return any(on_date)

class MeanStepsDataset(MinuteLevelActivtyDataset):
    def __init__(self,*args,**kwargs):
        super(MeanStepsDataset, self).__init__(*args, **kwargs)
        participant_ids = self.minute_data.index.get_level_values(level=0)
        dates = self.minute_data.index.get_level_values(level=1).date
        self.mean_steps = self.minute_data.groupby([participant_ids,dates])["steps"].sum().mean()

    def get_label(self,participant_id,start_date,end_date):
        participant_data = self.minute_data.loc[participant_id]
        return participant_data[participant_data.index.date == start_date.date()]["steps"].sum() > self.mean_steps        

class AutoencodeDataset(MinuteLevelActivtyDataset):
    def __init__(self,*args,**kwargs):
        super(AutoencodeDataset, self).__init__(*args, **kwargs)

    def get_label(self,participant_id,start_date,end_date):
        return None

    @lru_cache(maxsize=None)
    def __getitem__(self,index):
        # Could cache this later
        raise NotImplementedError

        # participant_id, end_date = self.participant_dates[index]
        # start_date = end_date - pd.Timedelta(self.day_window_size, unit = "days")
        # minute_data = self.get_user_data_in_time_range(participant_id,start_date,end_date)
        

        # if self.time_encoding == "sincos":
        #     minute_data["sin_time"]  = sin_time(minute_data.index)
        #     minute_data["cos_time"]  = cos_time(minute_data.index)

        # if self.return_dict:
        #     return {"inputs_embeds":minute_data.values.astype(np.float32),
        #             "labels":label}
        # return minute_data.values, label

def sin_time(timestamps):
    return np.sin(2*np.pi*(timestamps.hour * 60 + timestamps.minute)/MIN_IN_DAY).astype(np.float32)

def cos_time(timestamps):
    return np.cos(2*np.pi*(timestamps.hour * 60 + timestamps.minute)/MIN_IN_DAY).astype(np.float32)

if __name__ == "__main__":
    
    lab_results_reader = LabResultsReader()
    participant_ids = lab_results_reader.participant_ids
    minute_level_reader = MinuteLevelActivityReader(participant_ids=participant_ids)

    dataset = MinuteLevelActivtyDataset(minute_level_reader, lab_results_reader,
                                        participant_dates = minute_level_reader.participant_dates)

