import pandas as pd
from pandas.core.computation.eval import eval as _eval

import src.data.task_datasets as td

class FluPosLabler(object):
    def __init__(self):

        self.lab_results_reader = td.LabResultsReader()
        self.results = self.lab_results_reader.results
        self.results["_date"] = pd.to_datetime(self.results["trigger_datetime"].dt.date)
        self.results["is_pos"] = self.results["result"] == "Detected"
        
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

class ClauseLabler(object):
    def __init__(self, survey_respones, clause):
        self.clause = clause
        self.survey_responses = survey_respones
        self.survey_responses["_date"] = self.survey_responses["timestamp"].dt.date
        self.survey_lookup = self.survey_responses\
                                 .reset_index()\
                                 .drop_duplicates(subset=["participant_id","_date"],keep="last")\
                                 .set_index(["participant_id","_date"])\
                                 .to_dict('index')

    def __call__(self,participant_id,start_date,end_date):
        on_date = self.survey_lookup.get((participant_id,end_date.normalize()),None)
        if on_date:
            result = _eval(self.clause,local_dict=on_date)
            return result
        else:
            return 0
