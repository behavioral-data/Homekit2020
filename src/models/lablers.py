import pandas as pd
from pandas.core.computation.eval import eval as _eval
from pandas.core.indexes.datetimes import date_range
from pyspark.sql.functions import window

import src.data.task_datasets as td


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
        return is_pos_on_date

    def get_positive_keys(self):
        return set([(x[0], x[1].normalize()) for x in self.result_lookup.keys()])

class DayOfWeekLabler(object):
    """ Returns true if `end_date` is on one of `days`"""
    def __init__(self,days):
        self.days = days

    def __call__(self,participant_id,start_date,end_date):
        return end_date.weekday() in self.days
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
        return is_pos_on_date
    
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
        return result

    def get_positive_keys(self):
        return set([(x[0], x[1].normalize()) for x in self.survey_lookup.keys()])

