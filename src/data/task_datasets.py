from datetime import datetime
import multiprocessing

from functools import  reduce
from operator import and_ as bit_and
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from distributed import Client
import dask
from dask.diagnostics import ProgressBar
import dask.dataframe as dd

from dask_ml.preprocessing import StandardScaler
import xgboost as xgb
dask.config.set({"distributed.comm.timeouts.connect": "60"})
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from methodtools import lru_cache

from src.utils import get_logger, read_yaml
logger = get_logger(__name__)

from src.data.utils import get_dask_df, load_processed_table
from src.models.features import get_feature_with_name
from tqdm import tqdm

MIN_IN_DAY = 24*60

def feature_generator_from_config_path(feature_config_path,return_meta=True):
    feature_config = read_yaml(feature_config_path)
    print(feature_config["feature_names"])
    feature_fns = [(name,get_feature_with_name(name)) for name in feature_config["feature_names"]]

    meta = {}
    def gen_features(partiton):
        result = {}
        for name, fn in feature_fns:
            result[name] = float(fn(partiton))
            meta[name] = float
        return pd.Series(result)
    
    if return_meta:
        return gen_features, meta
    else:
        return gen_features

def calculate_daily_features_with_dask(feature_config_path):
    # Could almost certainly be improved with: https://docs.dask.org/en/latest/dataframe-groupby.html#aggregate
    
    feature_gen, meta  = feature_generator_from_config_path(feature_config_path)
    dask_df = get_dask_df("processed_fitbit_minute_level_activity")
    return dask_df.groupby(["participant_id","date"]).apply(feature_gen, result_type='expand', 
                                                            meta=meta).compute()

class DayLevelActivityReader(object):
    def __init__(self, min_date=None,
                       split_date=None,
                       max_date=None,
                       participant_ids=None,
                       day_window_size=15,
                       scaler=MinMaxScaler,
                       max_missing_days_in_window=5,
                       min_windows=1,
                       add_features_path=None,
                       data_location=None):
        
        self.min_date = min_date
        self.split_date = split_date
        self.max_missing_days_in_window = max_missing_days_in_window
        self.scaler = scaler
        self.min_windows = min_windows

        self.day_window_size = day_window_size
        self.max_missing_days_in_window = max_missing_days_in_window
        self.obs_per_day = 1

        df = load_processed_table("fitbit_day_level_activity", path=data_location)

        date_filters = []
        filters = []

        if min_date: 
            min_datetime = pd.to_datetime(min_date)
            date_filters.append(df["date"] >= min_datetime)
        
        if max_date:
            max_datetime = pd.to_datetime(max_date)
            date_filters.append(df["date"] < max_datetime)
        
        if date_filters:
            filters = reduce(bit_and,date_filters)
            
        if not len(filters) == 0:
            df = df[filters]
        
        if not participant_ids is None:
            df = df[df["participant_id"].isin(participant_ids)]
        
        # Need to add metadata to get to run only once
        valid_participant_dates = df.groupby("participant_id").apply(self.get_valid_dates)
        self.daily_data = df.set_index(["participant_id","date"])

        self.participant_dates = list(valid_participant_dates.dropna().apply(pd.Series).stack().droplevel(-1).items())
        
        if add_features_path:
            features = pd.read_csv(add_features_path)
            features["date"] = pd.to_datetime(features["date"])
            features = features.set_index(["participant_id","date"])
            self.daily_data = self.daily_data.join(features,how="left")

        if self.scaler:
            scale_model = self.scaler()
            self.daily_data[self.daily_data.columns] = scale_model.fit_transform(self.daily_data)
        
        self.activity_data = self.daily_data.sort_index()


    def get_valid_dates(self, partition):
        dates_with_data = pd.DatetimeIndex(partition[~partition["missing_hr"].astype(bool)]["date"].dt.date.unique())
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

    def split_participant_dates(self,date=None,eval_frac=None, by_participant=False,
                                limit_train_frac=False):
            """If random, split a fraction equal to random for eval,
                else, split participant dates to be before and after a given date"""

            if eval_frac:
                if by_participant:
                    ids = [x[0] for x in self.participant_dates]
                    all_ids = list(set(ids))
                    left_ids = all_ids[:int(len(all_ids)*eval_frac)]
                    if limit_train_frac:
                        left_ids = left_ids[:int(limit_train_frac*len(left_ids))]
                    
                    left_ids = set(left_ids)
                    train = [x for x in self.participant_dates if x[0] in left_ids]
                    eval = [x for x in self.participant_dates if not x[0] in left_ids]
                else:
                    train,eval = train_test_split(self.participant_dates,test_size=eval_frac)
                    train = train[:int(limit_train_frac*len(train))]
            elif date:
                if limit_train_frac:
                    raise NotImplementedError
                train = [x for x in self.participant_dates if x[1] <= pd.to_datetime(date) ]
                eval = [x for x in self.participant_dates if x[1] > pd.to_datetime(date) ]
            else:
                raise ValueError("If splitting, must either provide a date or fraction")
            return train, eval

class MinuteLevelActivityReader(object):
    def __init__(self, min_date=None,
                       split_date=None,
                       max_date=None,
                       participant_ids=None,
                       day_window_size=15,
                       scaler=StandardScaler,
                       max_missing_days_in_window=5,
                       min_windows=1,
                       data_location=None,
                       **_):
        
        self.min_date = min_date
        self.max_date = max_date
        self.min_windows = min_windows
        self.scaler = scaler
        self.day_window_size = day_window_size
        self.max_missing_days_in_window = max_missing_days_in_window
        self.min_windows = min_windows
        self.obs_per_day = 24*60

        n_cores = multiprocessing.cpu_count()
        pbar = ProgressBar()
        pbar.register()
        
        #pylint:disable=unused-variable 
        with Client(n_workers=min(n_cores,16)) as client:
            dask_df = get_dask_df("processed_fitbit_minute_level_activity",
                                    path = data_location)
    
            if not participant_ids is None:
                dask_df = dask_df[dask_df["participant_id"].isin(participant_ids)] 

            date_filters = []
            filters = []

            if min_date: 
                min_datetime = pd.to_datetime(min_date)
                date_filters.append(dask_df["timestamp"] >= min_datetime)
            
            if max_date:
                max_datetime = pd.to_datetime(max_date)
                date_filters.append(dask_df["timestamp"] < max_datetime)
            
            if date_filters:
                filters = reduce(bit_and,date_filters)

            if not len(filters) == 0:
                dask_df = dask_df[filters]
            
            self.day_window_size = day_window_size
            self.max_missing_days_in_window = max_missing_days_in_window
            self.min_windows = min_windows
            

            valid_participant_dates = dask_df.groupby("participant_id").apply(self.get_valid_dates, meta=("dates",object))
            
            self.activity_data = dask_df
        
            logger.info("Using Dask to pre-process data:")
            if self.scaler:
                logger.info("Scaling Data...")
                scale_model = self.scaler()
                numeric = self.activity_data.select_dtypes(include=['float64'])
                self.activity_data[list(numeric.columns.values)] = scale_model.fit_transform(numeric)

            valid_participant_dates, self.activity_data = dask.compute(valid_participant_dates,self.activity_data)
            logger.info("Setting index...")
            self.activity_data = self.activity_data.set_index(["participant_id","timestamp"]).drop(columns=["date"])
            self.participant_dates = list(valid_participant_dates.dropna().apply(pd.Series).stack().droplevel(-1).items())
        
       
            
        self.activity_data = self.activity_data.astype(np.float32)
        if not self.activity_data.index.is_monotonic:
            logger.info("Sorting Index...")
            self.activity_data = self.activity_data.sort_index()
            
    def get_valid_dates(self, partition):
        dates_with_data = pd.DatetimeIndex(partition[~partition["missing_heartrate"]]["timestamp"].dt.date.unique())
        min_days_with_data = self.day_window_size - self.max_missing_days_in_window
    
        if len(dates_with_data) < min_days_with_data:
            return 
        min_date = dates_with_data.min()
        max_date = dates_with_data.max()

        all_possible_start_dates = pd.date_range(min_date,max_date-pd.Timedelta(days=self.day_window_size-1))
        all_possible_end_dates = all_possible_start_dates + pd.Timedelta(days=self.day_window_size-1)

        min_days_with_data = self.day_window_size - self.max_missing_days_in_window

        mask = []
        for a,b in zip(all_possible_start_dates, all_possible_end_dates):
            has_enough_data = len(dates_with_data[(dates_with_data >= a) & (dates_with_data <= b) ] )>= min_days_with_data
            mask.append(has_enough_data)

        return all_possible_end_dates[mask].rename("dates")

    def split_participant_dates(self,date=None,eval_frac=None, by_participant=False,
                            limit_train_frac=False):
        """If random, split a fraction equal to random for eval,
            else, split participant dates to be before and after a given date"""

        if eval_frac:
            if by_participant:
                ids = [x[0] for x in self.participant_dates]
                all_ids = list(set(ids))
                left_ids = set(all_ids[:int(len(all_ids)*eval_frac)])
                if limit_train_frac:
                    left_ids = left_ids[:int(limit_train_frac*len(left_ids))]

                train = [x for x in self.participant_dates if x[0] in left_ids]
                eval = [x for x in self.participant_dates if not x[0] in left_ids]
            else:
                train,eval = train_test_split(self.participant_dates,test_size=eval_frac)
                train = train[:int(limit_train_frac*len(train))]
        elif date:
            if limit_train_frac:
                raise NotImplementedError
            train = [x for x in self.participant_dates if x[1] <= pd.to_datetime(date) ]
            eval = [x for x in self.participant_dates if x[1] > pd.to_datetime(date) ]
        else:
            raise ValueError("If splitting, must either provide a date or fraction")
        return train, eval

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

class ActivtyDataset(Dataset):
    def __init__(self, activity_reader,
                       lab_results_reader,
                       participant_dates,
                       max_missing_days_in_window=5,
                       min_windows=1,
                       scaler = MinMaxScaler,
                       time_encoding=None,
                       return_dict=False,
                       return_global_attention_mask = False,
                       add_absolute_embedding = False,
                       add_cls=False,
                       shuffle=False,
                       cache=True,
                       **_):
        
        self.activity_reader = activity_reader
        self.activity_data = self.activity_reader.activity_data
        self.activity_data = self.activity_data.sort_index()
        self.day_window_size = self.activity_reader.day_window_size 

        self.lab_results_reader = lab_results_reader
        
        self.participant_dates = participant_dates
        if shuffle:
            random.shuffle(self.participant_dates)
        
        self.time_encoding = time_encoding
        self.add_absolute_embedding = add_absolute_embedding

        self.return_dict = return_dict
        self.return_global_attention_mask = return_global_attention_mask
        
        n_features = self.activity_reader.activity_data.shape[-1] + 2*bool(time_encoding)
        
        obs_per_day = self.activity_reader.obs_per_day
        n_timesteps = (obs_per_day*self.day_window_size + int(add_cls))
        self.size = (n_timesteps, n_features)

        self.add_cls = add_cls

        if self.add_cls:
            self.cls_init = np.random.randn(1,self.size[-1]).astype(np.float32)   

        self.cache = cache
        if self.cache:
            self.get_item_cache = {}

    def get_user_data_in_date_range(self,participant_id, start, end):
        eod = end + pd.to_timedelta("1D") - pd.to_timedelta("1min")
        return self.get_user_data_in_time_range(participant_id,start,eod)

    def get_user_data_in_time_range(self,participant_id,start,end):
        assert self.activity_data.index.is_unique
        assert self.activity_data.index.is_monotonic

        if isinstance(self.activity_reader,DayLevelActivityReader):
            return self.activity_data.loc[(participant_id,start):(participant_id,end)]
        else:
            start_ix = self.activity_data.index.get_loc((participant_id,start))
            end_ix = self.activity_data.index.get_loc((participant_id,end)) + 1
            return self.activity_data.iloc[start_ix:end_ix]

    def get_label(self,participant_id,start_date,end_date):
        raise NotImplementedError
        try:
            participant_results = self.lab_results_reader.results.loc[participant_id]
        except KeyError:
            return None
        
        # Indexing returns a series when there's only one result
        if type(participant_results) == pd.Series:
            participant_results = participant_results.to_frame().T
        
        on_date = participant_results["trigger_datetime"].dt.date == end_date.date()
        is_pos = participant_results["result"] == "Detected"
        return any(on_date & is_pos)

    def __len__(self):
        return len(self.participant_dates)
    
    def __getitem__(self,index):
        # Could cache this later
        if self.cache:
            try:
                return self.get_item_cache[index]
            except KeyError:
                pass

        participant_id, end_date = self.participant_dates[index]
        start_date = end_date - pd.Timedelta(self.day_window_size -1, unit = "days")
        activity_data = self.get_user_data_in_date_range(participant_id,start_date,end_date)
        
        result = self.get_label(participant_id,start_date,end_date)
        if result:
            label = 1
        else:
            label = 0
        
        if self.time_encoding == "sincos":
            activity_data["sin_time"]  = sin_time(activity_data.index)
            activity_data["cos_time"]  = cos_time(activity_data.index)
        
        # activity_data = activity_data.T

        if self.add_absolute_embedding:
            activity_data = activity_data + sinu_position_encoding(*self.size)
        
        if self.return_dict:
            
            item = {}
            embeds = activity_data.values
            
            if self.add_cls:
                embeds = np.concatenate([self.cls_init,embeds],axis=0)
            
            item["inputs_embeds"] = embeds
            item["label"] = label

            if self.return_global_attention_mask:
                mask = np.zeros(embeds.shape[0])
                mask[0] = 1
                item["global_attention_mask"] = mask
            
            result = item
        else:
            result = activity_data.values, label

        if self.cache:
            self.get_item_cache[index] = result
        
        return result
    
    def to_stacked_numpy(self,flatten = False):
        
        if len(self)==0:
            return np.array([]), np.array([])
        X = []
        y = []

        for el_x, el_y in tqdm(self, desc = "Converting to np Array"): 
            if flatten:
                el_x = el_x.flatten()
    
            X.append(el_x)
            y.append(el_y)
        
        return np.stack(X), np.stack(y)

    def to_dmatrix(self,flatten=True):
        X,y = self.to_stacked_numpy(flatten=flatten)
        feature_names = self.get_feature_names()
        n_days = int(X.shape[-1] / len(feature_names))
        day_names = []
        for i in range(n_days):
            for name in feature_names:
                day_names.append(f"T-minus-{n_days-i-1}-{name}")
        return xgb.DMatrix(X,label=y, feature_names=day_names)

    def get_feature_names(self):
        names =list(self.activity_data.columns.values)
        if self.time_encoding == "sincos":
            names = names + ["sin_time","cos_time"]
        return names

class EarlyDetectionDataset(ActivtyDataset):
    def __init__(self,*args,**kwargs):
        super(EarlyDetectionDataset, self).__init__(*args, **kwargs)

class PredictTriggerDataset(ActivtyDataset):
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

class MeanStepsDataset(ActivtyDataset):
    def __init__(self,*args,**kwargs):
        super(MeanStepsDataset, self).__init__(*args, **kwargs)
        participant_ids = self.activity_data.index.get_level_values(level=0)
        dates = self.activity_data.index.get_level_values(level=1).date
        self.mean_steps = self.activity_data.groupby([participant_ids,dates])["steps"].sum().mean()

    def get_label(self,participant_id,start_date,end_date):
        participant_data = self.activity_data.loc[participant_id]
        return participant_data[participant_data.index.date == start_date.date()]["steps"].sum() > self.mean_steps        

class CustomLabler(ActivtyDataset):
    def __init__(self,*args,**kwargs):
        self.labeler = kwargs.pop("labeler")
        super(CustomLabler, self).__init__(*args, **kwargs)
    def get_label(self, participant_id, start_date, end_date):
        return self.labeler(participant_id,start_date,end_date)

class AutoencodeDataset(ActivtyDataset):
    def __init__(self,*args,**kwargs):
        super(AutoencodeDataset, self).__init__(*args, **kwargs)

    def get_label(self,participant_id,start_date,end_date):
        return None

    @lru_cache(maxsize=None)
    def __getitem__(self,index):
        # Could cache this later
        participant_id, end_date = self.participant_dates[index]
        start_date = end_date - pd.Timedelta(self.day_window_size, unit = "days")
        activity_data = self.get_user_data_in_date_range(participant_id,start_date,end_date)
        

        if self.time_encoding == "sincos":
            activity_data["sin_time"]  = sin_time(activity_data.index)
            activity_data["cos_time"]  = cos_time(activity_data.index)

        if self.return_dict:
            return {"inputs_embeds":activity_data.values.astype(np.float32),
                    "labels":activity_data.values.astype(np.float32)}
        
        return activity_data.values.astype(np.float32)

    def to_stacked_numpy(self):
        if len(self)==0:
            return np.array([]), np.array([])
        
        X = []
        for el_x in tqdm(self, desc = "Converting to np Array"): 
            X.append(el_x)

        return np.stack(X)

def sin_time(timestamps):
    return np.sin(2*np.pi*(timestamps.hour * 60 + timestamps.minute)/MIN_IN_DAY).astype(np.float32)

def cos_time(timestamps):
    return np.cos(2*np.pi*(timestamps.hour * 60 + timestamps.minute)/MIN_IN_DAY).astype(np.float32)

def sinu_position_encoding(n_position, d_pos_vec):
    '''Return the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2*i/d_pos_vec) for i in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    
    return position_enc

if __name__ == "__main__":
    
    lab_results_reader = LabResultsReader()
    participant_ids = lab_results_reader.participant_ids
    minute_level_reader = MinuteLevelActivityReader(participant_ids=participant_ids)

    dataset = ActivtyDataset(minute_level_reader, lab_results_reader,
                                        participant_dates = minute_level_reader.participant_dates)

