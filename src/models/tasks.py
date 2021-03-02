import src.data.task_datasets as td

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

    def __init__(self,split_date,dataset_args={}):
        lab_results_reader = td.LabResultsReader()
        participant_ids = lab_results_reader.participant_ids
        minute_level_reader = td.MinuteLevelActivityReader(participant_ids=participant_ids)

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

    
    