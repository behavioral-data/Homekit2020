from email.policy import default
import pandas as pd
import numpy as np

from pandas.core.computation.eval import eval as _eval
from pandas.core.indexes.datetimes import date_range
from pyspark.sql.functions import window

import src.data.task_datasets as td
from src.data.utils import load_processed_table


def get_dates_around(date,days_minus,days_plus):
    return pd.date_range(date + pd.to_timedelta(days_minus,unit="D"),
                         date + pd.to_timedelta(days_plus,unit="D"))

class FluPosLabler(object):
    def __init__(self, window_onset_min = 0,
                       window_onset_max = 0):

        self.lab_results_reader = td.LabResultsReader()
        self.results = self.lab_results_reader.results
        self.results["_date"] = pd.to_datetime(self.results["trigger_datetime"].dt.date)
        self.results["is_pos"] = self.results["result"] == "Detected"
        
        self.results = self.results[self.results["is_pos"]]
        mapper = lambda x: get_dates_around(x, days_minus=window_onset_min, days_plus=window_onset_max)
        self.results["_date"] = self.results["_date"].map(mapper)
        self.results = self.results.explode("_date")                                                        
        flus = ["Influenza A (Flu A)","Influenza B (Flu B)"]
        self.results = self.results[self.results["test_name"].isin(flus)]

        self.result_lookup = self.results\
                                 .reset_index()\
                                 .groupby(["participant_id","_date"])\
                                 ["is_pos"].any()\
                                 .to_dict()
                                 
    def __call__(self,participant_id,start_date,end_date):
        is_pos_on_date = self.result_lookup.get((participant_id,end_date.normalize()),False)
        return int(is_pos_on_date)

    def get_positive_keys(self):
        return set([(x[0], x[1].normalize()) for x in self.result_lookup.keys()])

class DailyFeaturesLabler(object):
    def __init__(self, window_size=7,
                       data_location=None,
                       normalize=True) -> None:

        super().__init__()
        self.df = load_processed_table("fitbit_day_level_activity", 
                                        path=data_location)
        feature_cols = ['resting_heart_rate',
                        'main_in_bed_minutes', 'main_efficiency', 'nap_count',
                        'total_asleep_minutes', 'total_in_bed_minutes', 'activityCalories',
                        'caloriesOut', 'caloriesBMR', 'marginalCalories', 'sedentaryMinutes',
                        'lightlyActiveMinutes', 'fairlyActiveMinutes', 'veryActiveMinutes',
                        'missing_hr', 'missing_sleep', 'missing_steps', 'missing_day']
        
        self.label_size = len(feature_cols) * window_size  
        if normalize:
            self.df[feature_cols] = (self.df[feature_cols]-self.df[feature_cols].mean())/self.df[feature_cols].std()
            
        self.df["features"] = self.df[feature_cols].to_numpy().tolist()
        self.result_lookup = {}

        n_non_missing = sum([not "missing" in x for x in feature_cols])
        
        self.default_features = np.array(([0]*n_non_missing  + [1]*(len(feature_cols)-n_non_missing)) * window_size).astype(np.float32)

        for user, group in iter(self.df.groupby("participant_id")):
            windows = list(group.set_index("date")["features"].rolling(f"{window_size}D",
                                                                              min_periods=window_size))
            for window in windows:
                if not len(window) == window_size:
                    continue

                concat_features = np.concatenate(window).astype(np.float32)
                self.result_lookup[(user,window.index[-1].normalize())] = concat_features
    
    def __call__(self,participant_id,start_date,end_date):
        return self.result_lookup.get((participant_id,end_date.normalize()),self.default_features)

class DayOfWeekLabler(object):
    """ Returns true if `end_date` is on one of `days`"""
    def __init__(self,days):
        self.days = days

    def __call__(self,participant_id,start_date,end_date):
        return end_date.weekday() in self.days

class AudereObeseLabler(object):
    def __init__(self):
        self.baseline_results = load_processed_table("baseline_screener_survey")\
                                                .drop_duplicates(subset=["participant_id"], 
                                                                  keep="last")\
                                                .set_index("participant_id")

        weight = self.baseline_results["weight"]
        height = (self.baseline_results["height__ft"] *12 + self.baseline_results["height__in"])
        is_obese = weight / (height**2) * 703 > 30.0
        self.results_lookup = is_obese.to_dict()


    def __call__(self,participant_id,start_date,end_date):
        return int(self.results_lookup.get(participant_id,False))

class EvidationILILabler(object):
    def __init__(self,feather_path,
                      ili_types=[1,2,3],
                      window_onset_min = -1,
                      window_onset_max = 5,
                      ):
        # Uses data like /projects/bdata/datasets/gatesfoundation/raw/UW-flu-label-transfer

        # LSFS and C19EX day-level labels
        """
        Datasets
        1. `lsfs-day-level-labels.feather` (N=33,950; rows=186,233; cols=5) - Large-scale Flu Survey (Fluvey2020) label set
        2. `c19ex-day-level-labels.feather` (N=4079; rows=1,843,238; cols=5) - COVID-19 Experiences (Covid2020) label set
        Use pandas `read_feather` to load these label sets

        ## Description
        * `participant_id` (string) and `date` (calendar date) can be used to match these labels to activity data
        * This approach uses a 43-day window around an "onset" date to create a unique event
        * A participant can have multple events (different onset dates) in the study
        * `ILI_type` (float) indicated the type of event:
            * `1` - generic ILI
            * `2` - medically-diagnosed Flu
            * `3` - symptomatic COVID-19
        * Each unique event for a participant can be identified via the column `event_number_v43` (float)
        * Each unique event has 43 days
            * -28 to +14 days around the event onset -- see `days_since_onset_v43` (float)
            * With `0` for the event onset date

        ## Column names
        1. `participant_id` (string) - participant ID to match activity data
        2. `date` (datetime) - calendar date to match activity data
        3. `event_number_v43` (float) - event idenfitier within each participant's labels
        4. `days_since_onset_v43` (float) - Each each (participant_id, event_number_v43) there are 43 values from -28 to 14
        5. `ILI_type` (float) - 1 (ILI), 2 (Flu), 3 (COVID-19)
        """
        self.results = pd.read_feather(feather_path)
        self.ili_types = ili_types

        self.results = self.results[self.results["ILI_type"].isin(self.ili_types)]
        self.results = self.results[self.results["days_since_onset_v43"].isin(list(range(window_onset_min,window_onset_max+1)))]
        self.results["is_pos"] = True
        
        self.result_lookup = self.results\
                                 .reset_index()\
                                 .groupby(["participant_id","date"])\
                                 ["is_pos"].any()\
                                 .to_dict()
                           
    def __call__(self,participant_id,start_date,end_date):
        is_pos_on_date = self.result_lookup.get((participant_id,end_date.normalize()),False)
        return int(is_pos_on_date)
    
    def get_positive_keys(self):
        return set([(x[0], x[1].normalize()) for x in self.result_lookup.keys()])


class CovidLabler(object):
    def __init__(self,path):
        """
        Designed for data like:
       "/projects/bdata/datasets/covid-fitbit/processed/covid_diagnosis_dates.csv"
        """
        self.results = pd.read_csv(path)
        self.results["covid_diagnosis_dates"] = pd.to_datetime(self.results["covid_diagnosis_dates"])
        self.result_lookup = (self.results\
                                 .reset_index()\
                                 .groupby(["participant_id","covid_diagnosis_dates"])\
                                 .size() > 0)\
                                 .to_dict()
                           
    def __call__(self,participant_id,start_date,end_date):
        is_pos_on_date = self.result_lookup.get((participant_id,end_date.normalize()),False)
        return int(is_pos_on_date)
    
    def get_positive_keys(self):
        return set([(x[0], x[1].normalize()) for x in self.result_lookup.keys()])



class ClauseLabler(object):
    def __init__(self, survey_respones, clause):
        self.clause = clause
        self.survey_responses = survey_respones
        self.survey_responses["_date"] = self.survey_responses["timestamp"].dt.normalize()
        self.survey_responses["_dummy"] = True
        self.survey_lookup = self.survey_responses\
                                 .reset_index()\
                                 .drop_duplicates(subset=["participant_id","_date"],keep="last")\
                                 .set_index(["participant_id","_date"])\
                                 .query(self.clause)\
                                 ["_dummy"]\
                                 .to_dict()

    def __call__(self,participant_id,start_date,end_date):
        result = self.survey_lookup.get((participant_id,end_date.normalize()),False)
        return int(result)

    def get_positive_keys(self):
        return set([(x[0], x[1].normalize()) for x in self.survey_lookup.keys()])

