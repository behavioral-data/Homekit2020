import sys
import datetime

import src.data.task_datasets as td
from src.models.eval import classification_eval, autoencode_eval
from src.data.utils import load_processed_table
from src.data.cache_datareader import load_cached_activity_reader
from src.utils import get_logger
logger = get_logger()

import pandas as pd

SUPPORTED_TASK_TYPES=[
    "classification",
    "autoencoder"
]
def get_task_with_name(name):
    try:
        identifier = getattr(sys.modules[__name__], name)
    except AttributeError:
        raise NameError(f"{name} is not a valid task.")
    if isinstance(identifier, type):
        return identifier
    raise TypeError(f"{name} is not a valid task.")

class Task(object):
    def __init__(self):
        for task_type in SUPPORTED_TASK_TYPES:
            setattr(self,f"is_{task_type}",False)
            
    def get_description(self):
        raise NotImplementedError
    def __str__(self):
        raise NotImplementedError
    
    def get_description(self):
        return self.__doc__

    def get_train_dataset(self):
        return self.train_dataset

    def get_eval_dataset(self):
        return self.eval_dataset

class ClassificationMixin():
    def __init__(self):
        self.is_classification = True

    def evaluate_results(self,logits,labels,threshold=0.5):
        return classification_eval(logits,labels,threshold=threshold)

    def get_huggingface_metrics(self,threshold=0.5):
        def evaluator(pred):
            labels = pred.label_ids
            logits = pred.predictions
            return self.evaluate_results(logits,labels,threshold=threshold)
        return evaluator

class AutoencodeMixin():
    def __init__(self):
        self.is_autoencoder = True

    def evaluate_results(self,preds,labels):
        return autoencode_eval(preds,labels)

    def get_huggingface_metrics(self):
        def evaluator(predictions):
            labels = predictions.label_ids
            preds = predictions.predictions
            return self.evaluate_results(preds,labels)
        return evaluator

class ActivityTask(Task):
    """ Is inhereited by anything that operatres over the minute
        level data"""
    def __init__(self,base_dataset,dataset_args={},
                     activity_level = "minute"):
        super(ActivityTask,self).__init__()
        
        split_date = dataset_args.pop("split_date",None)
        eval_frac = dataset_args.pop("eval_frac",None)

        if not split_date:
            raise KeyError("Must provide a date for splitting train and ")
        
        min_date = dataset_args.get("min_date",None)
        max_date = dataset_args.get("max_date",None)
        day_window_size = dataset_args.get("day_window_size",None)
        max_missing_days_in_window = dataset_args.get("max_missing_days_in_window",None)

        lab_results_reader = td.LabResultsReader()
        participant_ids = lab_results_reader.participant_ids

        if activity_level == "minute":
            base_activity_reader = td.MinuteLevelActivityReader
        else:
            base_activity_reader = td.DayLevelActivityReader

        if dataset_args.get("is_cached"):
            activity_reader = load_cached_activity_reader(self.get_name(),
                                                          dataset_args=dataset_args,
                                                          fail_if_mismatched=True)
        else:
            activity_reader = base_activity_reader(participant_ids=participant_ids,
                                                           min_date = min_date,
                                                           max_date = max_date,
                                                           day_window_size=day_window_size,
                                                           max_missing_days_in_window=max_missing_days_in_window)

        train_participant_dates, eval_participant_dates = activity_reader.split_participant_dates(date=split_date,eval_frac=eval_frac)
    

        self.train_dataset = base_dataset(activity_reader, lab_results_reader,
                        participant_dates = train_participant_dates,**dataset_args)
        self.eval_dataset = base_dataset(activity_reader, lab_results_reader,
                        participant_dates = eval_participant_dates,**dataset_args)
    
    def get_description(self):
        return self.__doc__

    def get_train_dataset(self):
        return self.train_dataset

    def get_eval_dataset(self):
        return self.eval_dataset
    
class GeqMeanSteps(ActivityTask, ClassificationMixin):
    """A dummy task to predict whether or not the total number of steps
       on the first day of a window is >= the mean across the whole dataset"""
    
    def __init__(self,dataset_args={}):
        ActivityTask.__init__(self,td.ActivtyDataset,dataset_args=dataset_args)
        ClassificationMixin.__init__(self)
        self.is_classification = True
    
    def evaluate_results(self,logits,labels,threshold=0.5):
        return classification_eval(logits,labels,threshold=threshold)
    
    def get_name(self):
        return "GeqMedianSteps"
    
    def get_huggingface_metrics(self,threshold=0.5):
        def evaluator(pred):
            labels = pred.label_ids
            logits = pred.predictions
            return self.evaluate_results(logits,labels,threshold=threshold)
        return evaluator

class PredictFluPos(ActivityTask, ClassificationMixin):
    """Predict the whether a participant was positive
       given a rolling window of minute level activity data.
       We validate on data after split_date, but before
       max_date, if provided"""

    def __init__(self,dataset_args={}, activity_level = "minute"):

        ActivityTask.__init__(self,td.ActivtyDataset,dataset_args=dataset_args,
                                 activity_level=activity_level)
        ClassificationMixin.__init__(self)
        

    def get_name(self):
        return "PredictFluPos"
    

class PredictTrigger(ActivityTask,ClassificationMixin):
    """Predict the whether a participant triggered the 
       test on the last day of a range of data"""

    def __init__(self,dataset_args={}, activity_level="minute"):
        ActivityTask.__init__(self,td.PredictTriggerDataset,dataset_args=dataset_args,
                               activity_level = "minute")
        ClassificationMixin.__init__(self)
        # self.is_classification = True

    def get_name(self):
        return "PredictTrigger"

class PredictSurveyCol(Task,ClassificationMixin):
    """Predict the whether a given column in the onehot
       encoded surverys is true for a given day"""

    def __init__(self,dataset_args={}):
        Task.__init__(self)
        ClassificationMixin.__init__(self)
        
        self.survey_col = dataset_args.pop("survey_col",None)
        if self.survey_col is None:
            raise ValueError("Must provide a column from 'daily_surveys_onehot.csv' as 'survey_col' in 'dataset_args'")
        
        split_date = dataset_args.pop("split_date",None)
        eval_frac = dataset_args.pop("eval_frac",None)

        if not split_date:
            raise KeyError("Must provide a date for splitting train and ")
        
        min_date = dataset_args.pop("min_date",None)
        max_date = dataset_args.pop("max_date",None)
        day_window_size = dataset_args.pop("day_window_size",None)
        max_missing_days_in_window = dataset_args.pop("max_missing_days_in_window",None)

        lab_results_reader = td.LabResultsReader()
        participant_ids = lab_results_reader.participant_ids

        
        minute_level_reader = td.MinuteLevelActivityReader(participant_ids=participant_ids,
                                                           min_date = min_date,
                                                           max_date = max_date,
                                                           day_window_size=day_window_size,
                                                           max_missing_days_in_window=max_missing_days_in_window)

        train_participant_dates, eval_participant_dates = minute_level_reader.split_participant_dates(date=split_date,eval_frac=eval_frac)

        survey_resutls = load_processed_table("lab_results_with_trigger_date")
        # has_survey = survey_resutls[survey_resutls[]]

    
        self.train_dataset = base_dataset(minute_level_reader, lab_results_reader,
                        participant_dates = train_participant_dates,**dataset_args)
        self.eval_dataset = base_dataset(minute_level_reader, lab_results_reader,
                        participant_dates = eval_participant_dates,**dataset_args)


    def get_name(self):
        return f"PredictSurveyCol-{self.survey_col}"

    def get_description(self):
        return self.__doc__

    def get_train_dataset(self):
        return self.train_dataset

    def get_eval_dataset(self):
        return self.eval_dataset

class SingleWindowActivityTask(Task):
    """Base class for tasks that make predictions about single windows of data
       (e.g.) predicting a user's BMI."""
    
    def __init__(self,base_dataset,dataset_args={}, activity_level="minute",
                window_selection="first"):
        Task.__init__(self)
        eval_frac = dataset_args.pop("eval_frac",None)
        split_date = dataset_args.pop("split_date",None)

        min_date = dataset_args.get("min_date",None)
        max_date = dataset_args.get("max_date",None)
        day_window_size = dataset_args.get("day_window_size",None)
        max_missing_days_in_window = dataset_args.get("max_missing_days_in_window",None)

        lab_results_reader = td.LabResultsReader()
        participant_ids = lab_results_reader.participant_ids

        if activity_level == "minute":
            base_activity_reader = td.MinuteLevelActivityReader
        else:
            base_activity_reader = td.DayLevelActivityReader

        if dataset_args.get("is_cached"):
            activity_reader = load_cached_activity_reader(self.get_name(),
                                                          dataset_args=dataset_args,
                                                          fail_if_mismatched=True)
        else:
            activity_reader = base_activity_reader(participant_ids=participant_ids,
                                                           min_date = min_date,
                                                           max_date = max_date,
                                                           day_window_size=day_window_size,
                                                           max_missing_days_in_window=max_missing_days_in_window)

        train_participant_dates, eval_participant_dates = activity_reader.split_participant_dates(date=split_date,eval_frac=eval_frac)
        if window_selection == "first":
            train_participant_dates = self.filter_participant_dates(train_participant_dates)
            eval_participant_dates = self.filter_participant_dates(eval_participant_dates)
    
        else:
            raise NotImplementedError


        self.train_dataset = base_dataset(activity_reader, lab_results_reader,
                        participant_dates = train_participant_dates,**dataset_args)
        self.eval_dataset = base_dataset(activity_reader, lab_results_reader,
                        participant_dates = eval_participant_dates,**dataset_args)
    
    def filter_participant_dates(self,participant_dates):
        candidates = {}
        for id, date in participant_dates:
            if candidates.get(id,datetime.datetime.max) > date:
                candidates[id] = date
        return list(candidates.items())

class ClassifyObese(SingleWindowActivityTask, ClassificationMixin):
    def __init__(self, dataset_args, activity_level, window_selection="first"):
        dataset_args["labeler"] = self.get_labeler()
        SingleWindowActivityTask.__init__(self, td.CustomLabler, 
                         dataset_args=dataset_args, 
                         activity_level=activity_level, 
                         window_selection=window_selection)
        ClassificationMixin.__init__(self)      

    def get_labeler(self):
        self.baseline_results = load_processed_table("baseline_screener_survey")
        def labeler(participant_id, start_date, end_date):
            participant_data = self.baseline_results[self.baseline_results["participant_id"]==participant_id].iloc[0]
            weight = participant_data["weight"]
            height = (participant_data["height__ft"] *12 + participant_data["height__in"])
            return weight / (height**2) * 703 > 30.0
        return labeler
    
    def get_name(self):
        return "ClassifyObese"
class EarlyDetection(ActivityTask):
    """Mimics the task used by Evidation Health"""

    def __init__(self,base_dataset=td.EarlyDetectionDataset,dataset_args={},
                      activity_level = "minute"):
        eval_frac = dataset_args.pop("eval_frac",None)

        if not eval_frac:
            raise KeyError("Must provide an eval fraction for splitting train and test")
        
        min_date = dataset_args.pop("min_date",None)
        max_date = dataset_args.pop("max_date",None)
        day_window_size = dataset_args.pop("day_window_size",4)
        window_pad = dataset_args.pop("window_pad",20)

        max_missing_days_in_window = dataset_args.pop("max_missing_days_in_window",None)

        lab_results_reader = td.LabResultsReader(pos_only=True)
        participant_ids = lab_results_reader.participant_ids

        if activity_level == "minute":
            base_activity_reader = td.MinuteLevelActivityReader
        else:
            base_activity_reader = td.DayLevelActivityReader
            
        if dataset_args.get("is_cached"):
            logger.info("Loading cached data reader...")
            activity_reader = load_cached_activity_reader(self.get_name())
        else:
            activity_reader = base_activity_reader(participant_ids=participant_ids,
                                                           min_date = min_date,
                                                           max_date = max_date,
                                                           day_window_size=day_window_size,
                                                           max_missing_days_in_window=max_missing_days_in_window)


        pos_dates = lab_results_reader.results
        pos_dates["timestamp"] = pos_dates["trigger_datetime"].dt.floor("D")
        delta = pd.to_timedelta(window_pad + day_window_size, unit = "days")
        
        label_date = list(pos_dates["timestamp"].items())
        after  = list((pos_dates["timestamp"] + delta).items())
        before  = list((pos_dates["timestamp"] -  delta).items())


        orig_valid_dates = set(activity_reader.participant_dates)
        new_valid_dates = list(set(label_date+after+before).intersection(orig_valid_dates))
        
        valid_participants = list(set([x[0] for x in new_valid_dates]))
        n_valid_participants = len(valid_participants)
        split_index = int((1-eval_frac) * n_valid_participants)

        train_participants = valid_participants[:split_index]
        eval_participants = valid_participants[split_index:]


        train_participant_dates = [x for x in new_valid_dates if x[0] in train_participants]
        eval_participant_dates = [x for x in new_valid_dates if x[0] in eval_participants]

        self.train_dataset = base_dataset(activity_reader, lab_results_reader,
                        participant_dates = train_participant_dates,**dataset_args)
        self.eval_dataset = base_dataset(activity_reader, lab_results_reader,
                        participant_dates = eval_participant_dates,**dataset_args)
        
        self.is_classification = True
    
    def get_name(self):
        return "EarlyDetection"
    
    def evaluate_results(self,logits,labels,threshold=0.5):
        return classification_eval(logits,labels,threshold=threshold)

    def get_huggingface_metrics(self,threshold=0.5):
        def evaluator(pred):
            labels = pred.label_ids
            logits = pred.predictions
            return self.evaluate_results(logits,labels,threshold=threshold)
        return evaluator

class AutoencodeEarlyDetection(AutoencodeMixin, EarlyDetection):
    """Autoencode minute level data"""

    def __init__(self,dataset_args={}):
        EarlyDetection.__init__(self,td.AutoencodeDataset,dataset_args=dataset_args)
        AutoencodeMixin.__init__(self)
        self.is_autoencoder = True

    def get_description(self):
        return self.__doc__
    
    def get_train_dataset(self):
        return self.train_dataset

    def get_eval_dataset(self):
        return self.eval_dataset
    
    def get_name(self):
        return "Autoencode"
    

class Autoencode(AutoencodeMixin, ActivityTask):
    """Autoencode minute level data"""

    def __init__(self,dataset_args={}):
        ActivityTask.__init__(self,td.AutoencodeDataset,dataset_args=dataset_args)
        AutoencodeMixin.__init__(self)
        self.is_autoencoder = True

    def get_description(self):
        return self.__doc__
    
    def get_train_dataset(self):
        return self.train_dataset

    def get_eval_dataset(self):
        return self.eval_dataset
    
    def get_name(self):
        return "Autoencode"
    