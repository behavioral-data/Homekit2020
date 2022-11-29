from email.policy import default
from re import L
import pandas as pd
import numpy as np

from pandas.core.computation.eval import eval as _eval
from pandas.core.indexes.datetimes import date_range
from pyspark.sql.functions import window

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import sqlite3


from src.data.utils import load_processed_table


def get_dates_around(date,days_minus,days_plus):
    return pd.date_range(date - pd.to_timedelta(days_minus,unit="D"),
                         date + pd.to_timedelta(days_plus,unit="D"))

#TODO eventually deprecate this
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
class FluPosLabler(object):
    def __init__(self, window_onset_min = 0,
                       window_onset_max = 0):

        self.lab_results_reader = LabResultsReader()
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


class FluPosWeakLabler(object):
    def __init__(self, survey_responses, data_location=None, window_size=7, normalize=True, window_onset_min = 0, window_onset_max = 0):

        """ survey responses """
        self.survey_responses = survey_responses.drop("Unnamed: 0", axis=1)
        self.survey_responses["_date"] = self.survey_responses["timestamp"].dt.normalize()

        """ lab results """
        flus = ["Influenza A (Flu A)","Influenza B (Flu B)"]
        self.results = LabResultsReader().results.drop("Unnamed: 0", axis=1)
        self.results["_date"] = pd.to_datetime(self.results["trigger_datetime"].dt.date)

        # import pdb; pdb.set_trace()

        activity = load_processed_table("fitbit_day_level_activity", path=data_location)
        activity["_date"] = pd.to_datetime(activity["date"].dt.date)
        del activity["date"]

        # import pdb; pdb.set_trace()
        feature_cols = ['resting_heart_rate', "body_temp_f",
                        'main_in_bed_minutes', 'main_efficiency', 'nap_count',
                        'total_asleep_minutes', 'total_in_bed_minutes', 'activityCalories',
                        'caloriesOut', 'caloriesBMR', 'marginalCalories', 'sedentaryMinutes',
                        'lightlyActiveMinutes', 'fairlyActiveMinutes', 'veryActiveMinutes',
                        'missing_hr', 'missing_sleep', 'missing_steps', 'missing_day']

        self.label_size = len(feature_cols) * window_size


        self.results["is_pos"] = (self.results["result"] == "Detected") & (self.results["test_name"].isin(flus))
        self.results = self.results.groupby(["participant_id", "_date", "first_report_yn"])["is_pos"].any().reset_index()
        self.results["flu_prob"] = 0.0
        self.results.loc[self.results["is_pos"] == True, "flu_prob"] = 1.0

        # import pdb; pdb.set_trace()
        # merges flu results to survey responses
        self.merged_table = self.survey_responses.merge(self.results, on=["participant_id", "_date", "first_report_yn"], how="left")
        self.merged_table = self.merged_table.merge(activity, on=["participant_id", "_date"], how="left")

        flu_features = np.concatenate((np.array(self.merged_table.columns[8:-2]), np.array(feature_cols)))
        # flu_features = self.merged_table.columns[8:-2]

        """ fits label model """
        labeled_responses = self.merged_table[~pd.isna(self.merged_table["is_pos"])]
        X_labeled = labeled_responses[flu_features]
        y_labeled = labeled_responses["is_pos"]
        lm, scaler = fit_labelmodel(X_labeled, y_labeled)

        """ initializes result table """
        # self.results = self.merged_table[(~pd.isna(self.merged_table["is_pos"])) | (self.merged_table["first_report_yn"])]


        """ adds label predictions to table """
        missing_response_idx = (pd.isna(self.merged_table["is_pos"])) & (self.merged_table["first_report_yn"]==True)
        X_missing = self.merged_table[missing_response_idx][flu_features]
        self.merged_table.loc[missing_response_idx, "flu_prob"] = predict_labelmodel(lm, X_missing, scaler)[:, 1]


        return_table = self.merged_table[~pd.isna(self.merged_table["flu_prob"])]

        mapper = lambda x: get_dates_around(x, days_minus=window_onset_min, days_plus=window_onset_max)
        return_table["_date"] = return_table["_date"].map(mapper)
        return_table = return_table.explode("_date")

        self.result_lookup = return_table \
            .reset_index() \
            .groupby(["participant_id","_date"]) \
            ["flu_prob"].max() \
            .to_dict()

    def _get_partition_results(self, results, file_path: str):
        partition = pd.read_csv(file_path)

        qry = '''
            SELECT *
            FROM results 
            WHERE EXISTS (
                SELECT 1 
                FROM partition
                WHERE 
                    ((results._date BETWEEN partition.start AND partition.end) 
                    AND 
                    (partition.participant_id = results.participant_id))
                )
            '''
        conn = sqlite3.connect(':memory:')
        results.reset_index().to_sql('results', conn, index=False)
        partition.to_sql('partition', conn, index=False)

        return pd.read_sql_query(qry, conn)


    def __call__(self,participant_id,start_date,end_date):
        flu_prob_on_date = self.result_lookup.get((participant_id,end_date.normalize()),False)
        return float(flu_prob_on_date)

    def get_positive_keys(self):
        return set([(x[0], x[1].normalize()) for x in self.result_lookup.keys()])

def float_mapper(x):
    try:
        return float(x)
    except:
        return -1

def fit_labelmodel(x, y):

    x = x.fillna(-1).applymap(float_mapper)

    x_train, x_val = train_test_split(x.to_numpy(), train_size=0.8)
    y_train, y_val = train_test_split(y.to_numpy().astype(int), train_size=0.8)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)


    # model = GradientBoostingClassifier(n_estimators=100)
    model = RandomForestClassifier(max_features="sqrt")

    model.fit(x_train, y_train)

    preds = model.predict_proba(x_val)

    fpr, tpr, thresholds = roc_curve(y_val, preds[:, 1])

    print("PR-AUC: {}".format(average_precision_score(y_val, preds[:, 1])))
    print("ROC-AUC: {}".format(auc(fpr, tpr)))

    return model, scaler


def predict_labelmodel(lm, x, scaler):
    x = scaler.transform(x.fillna(-1).applymap(float_mapper))
    return lm.predict_proba(x)


class CleanAnnotationLabler(object):
    def __init__(self, survey_responses, window_onset_min = 0,
                 window_onset_max = 0):

        self.survey_responses = survey_responses
        self.survey_responses["_date"] = self.survey_responses["timestamp"].dt.normalize()

        groups = (self.survey_responses["_date"].diff() != pd.Timedelta("1d")).cumsum()
        self.survey_responses["response_groups"] = groups

        flus = ["Influenza A (Flu A)","Influenza B (Flu B)"]
        self.lab_results_reader = LabResultsReader()
        self.results = self.lab_results_reader.results
        self.results["_date"] = pd.to_datetime(self.results["trigger_datetime"].dt.date)
        self.results["is_pos"] = (self.results["result"] == "Detected") & (self.results["test_name"].isin(flus))

        self.merged_table = self.survey_responses.merge(self.results, on=["participant_id", "_date", "first_report_yn"], how="left")

        # if a participant tested positive at any given time,
        # all symptomatic consecutive days during the same time period are considered positive
        dd = self.merged_table.groupby("response_groups")["is_pos"].apply(lambda x: x.any() if not x.isnull().all() else np.nan).to_dict()
        self.merged_table["is_pos"] = self.merged_table["response_groups"].apply(lambda d: dd[d])


        mapper = lambda x: get_dates_around(x, days_minus=window_onset_min, days_plus=window_onset_max)
        self.merged_table["_date"] = self.merged_table["_date"].map(mapper)
        self.merged_table = self.merged_table.explode("_date")

        # participant exhibits flu-like symptoms but is missing a PCR report
        self.merged_table["is_missing"] = ~(((self.merged_table["have_flu"]==1) & ~pd.isna(self.merged_table["is_pos"])) | (self.merged_table["have_flu"]==0))
        missing = self.merged_table[self.merged_table["is_missing"]==True]

        self.result_lookup = missing \
            .reset_index() \
            .groupby(["participant_id","_date"]) \
            ["is_missing"].any() \
            .to_dict()

    def __call__(self,participant_id,start_date,end_date):
        is_clean = not self.result_lookup.get((participant_id,end_date.normalize()),False)
        return int(is_clean)

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


class CovidSignalsLabler(object):
    def __init__(self, window_onset_min: int = 0,
                       window_onset_max: int = 0):
        """
        Designed for operation with the "Covid Signals" dataset
        """
        self.lab_results = pd.read_csv("data/processed/COVID_Signals/labs.csv",
                                      dtype={"id_participant_external":"str"})
        self.lab_results["date_received"]=pd.to_datetime(self.lab_results["samples_received"],utc=True).astype('<M8[ns]')

        self.weekly_surveys = pd.read_csv("data/processed/COVID_Signals/weekly_surveys.csv",
                                          dtype={"id_participant_external":"str"})
        self.weekly_surveys["date_completed"] = pd.to_datetime(self.weekly_surveys["date_of_test_kit_completion"].str.split("|").str[-1],errors="coerce")
        self.weekly_surveys = self.weekly_surveys.dropna(subset=["date_completed"])
    
        self.results_and_surveys = pd.merge_asof(self.lab_results.sort_values("date_received"),
                                                 self.weekly_surveys.sort_values("date_completed"),
                                                 by="id_participant_external",
                                                 left_on="date_received",
                                                 right_on="date_completed",
                                                 tolerance=pd.to_timedelta("7D"),
                                                 direction="backward")
        #TODO: Where do these NaT values come from?
        self.positive_results = self.results_and_surveys[self.results_and_surveys["result"]=="Positive"].dropna(subset = "date_completed")
        mapper = lambda x: get_dates_around(x, days_minus=window_onset_min, days_plus=window_onset_max)
        self.positive_results["date_completed"] = self.positive_results["date_completed"].map(mapper)
        self.positive_results = self.positive_results.explode("date_completed")  

        self.result_lookup = (self.positive_results\
                                 .reset_index()\
                                 .groupby(["id_participant_external","date_completed"])\
                                 .size() > 0)\
                                 .to_dict()
        self.pids = set()
                           
    def __call__(self,participant_id,start_date,end_date):
        is_pos_on_date = self.result_lookup.get((participant_id,end_date.normalize()),False)
        self.pids.add(participant_id)
        return int(is_pos_on_date)

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


class SameParticipantLabler(object):
    def __init__(self):
        return 

    def __call__(self,participant_id_l,start_l,end_l,
                      participant_id_r,start_r,end_r):
        return int(participant_id_l == participant_id_r)

class SequentialLabler(object):
    def __init__(self):
        return 

    def __call__(self,participant_id_l,start_l,end_l,
                      participant_id_r,start_r,end_r):
        correct_date = (start_r - end_l).days == 0 
        correct_pid = participant_id_l == participant_id_r
        label = correct_date and correct_pid
        return int(label)
