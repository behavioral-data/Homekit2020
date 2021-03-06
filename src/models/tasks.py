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


class PredictFluPos(Task):
    """Predict the whether a participant was positive
       given a rolling window of minute level activity data.
       We validate on data after split_date, but before
       max_date, if provided"""

    def __init__(self,dataset_args={}):
        split_date = dataset_args.pop("split_date",None)
        if not split_date:
            raise KeyError("Must provide a date for splitting train and ")
        
        min_date = dataset_args.pop("min_date",None)
        max_date = dataset_args.pop("max_date",None)

        lab_results_reader = td.LabResultsReader()
        participant_ids = lab_results_reader.participant_ids

        minute_level_reader = td.MinuteLevelActivityReader(participant_ids=participant_ids,
                                                           min_date = min_date,
                                                           max_date = max_date)

        train_participant_dates, eval_participant_dates = minute_level_reader.split_participant_dates(split_date)
    

        self.train_dataset = td.MinuteLevelActivtyDataset(minute_level_reader, lab_results_reader,
                        participant_dates = train_participant_dates,**dataset_args)
        self.eval_dataset = td.MinuteLevelActivtyDataset(minute_level_reader, lab_results_reader,
                        participant_dates = eval_participant_dates,**dataset_args)

        
    def get_description(self):
        return self.__doc__

    def get_train_dataset(self):
        return self.train_dataset

    def get_eval_dataset(self):
        return self.eval_dataset
    
    def get_name(self):
        return "PredictFluPos"
    
    def evaluate_results(self,logits,labels):
        return classification_eval(logits,labels)
    
    