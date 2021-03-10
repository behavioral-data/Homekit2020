import sys

import src.data.task_datasets as td
from src.models.eval import classification_eval


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
        pass
    def get_description(self):
        raise NotImplementedError
    def __str__(self):
        raise NotImplementedError

class MinuteLevelTask(Task):
    """ Is inhereited by anything that operatres over the minute
        level data"""
    def __init__(self,dataset_args={}):
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
    

        self.train_dataset = td.MeanStepsDataset(minute_level_reader, lab_results_reader,
                        participant_dates = train_participant_dates,**dataset_args)
        self.eval_dataset = td.MeanStepsDataset(minute_level_reader, lab_results_reader,
                        participant_dates = eval_participant_dates,**dataset_args)

class GeqMeanSteps(MinuteLevelTask):
    """A dummy task to predict whether or not the total number of steps
       on the first day of a window is >= the mean across the whole dataset"""
    
    def __init__(self,dataset_args={}):
        super(GeqMeanSteps,self).__init__(dataset_args=dataset_args)

    def get_train_dataset(self):
        return self.train_dataset

    def get_eval_dataset(self):
        return self.eval_dataset
    
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

class PredictFluPos(MinuteLevelTask):
    """Predict the whether a participant was positive
       given a rolling window of minute level activity data.
       We validate on data after split_date, but before
       max_date, if provided"""

    def __init__(self,dataset_args={}):
        super(GeqMeanSteps,self).__init__(dataset_args=dataset_args)
        
    def get_description(self):
        return self.__doc__

    def get_train_dataset(self):
        return self.train_dataset

    def get_eval_dataset(self):
        return self.eval_dataset
    
    def get_name(self):
        return "PredictFluPos"
    
    def evaluate_results(self,logits,labels,threshold=0.5):
        return classification_eval(logits,labels,threshold=threshold)

    def get_huggingface_metrics(self,threshold=0.5):
        def evaluator(pred):
            labels = pred.label_ids
            logits = pred.predictions
            return self.evaluate_results(logits,labels,threshold=threshold)
        return evaluator


class Autoencode(MinuteLevelTask):
    pass