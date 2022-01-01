"""
===========================
SeattleFluStudy experiments  
===========================
`Project repository available here  <https://github.com/behavioral-data/SeattleFluStudy>`_

This module contains the code used in the experimental tasks described in the reference paper.
The tasks are classes intended to provide, during training and evaluation, the correct reference to 
the datasets being used and evaluation metrics. 

**Classes**
    :class Task:
    :class ClassificationMixin:
    :class AutoencodeMixin:
    :class ActivityTask: 
    :class GeqMeanSteps: 
    :class PredictFluPos: 
    :class PredictEvidationILI: 
    :class PredictTrigger:
    :class PredictSurveyClause:
    :class SingleWindowActivityTask:
    :class ClassifyObese:
    :class EarlyDetection:
    :class AutoencodeEarlyDetection:
    :class Autoencode:

"""
__docformat__ = 'reStructuredText'

import sys
import datetime
from warnings import WarningMessage
import os

from pyarrow.parquet import ParquetDataset
from sklearn.utils import resample

from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator
import ray

import numpy as np

from petastorm import make_reader
from petastorm.transform import TransformSpec
from petastorm.etl.dataset_metadata import infer_or_load_unischema
import petastorm.predicates  as peta_pred

import src.data.task_datasets as td
from src.models.eval import classification_eval, autoencode_eval
from src.data.utils import load_processed_table, load_cached_activity_reader, url_from_path
from src.utils import get_logger, read_yaml
from src.models.lablers import FluPosLabler, ClauseLabler, EvidationILILabler, DayOfWeekLabler, AudereObeseLabler

logger = get_logger(__name__)

import pandas as pd


###################################################
########### MODULE UTILITY FUNCTIONS ##############
###################################################

SUPPORTED_TASK_TYPES=[      
    "classification",
    "autoencoder"
]
def get_task_with_name(name): 
    """
    Checks the provided `name` is within the module definitions, raises `NameError` if it isn't
       :returns: an object referencing the desired task (specified in input)
    """
    try:
        identifier = getattr(sys.modules[__name__], name) # get a reference to the module itself through the sys.modules dictionary  
    except AttributeError:
        raise NameError(f"{name} is not a valid task.")
    if isinstance(identifier, type):
        return identifier
    raise TypeError(f"{name} is not a valid task.")


def get_task_from_config_path(path,**kwargs):
    config = read_yaml(path)
    task_class = get_task_with_name(config["task_name"])
    task = task_class(dataset_args=config.get("dataset_args",{}),
                      **config.get("task_args",{}),                      
                      **kwargs)
    return task

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
    
    def get_labler(self):
        return NotImplementedError
    
    def get_train_dataloader(self,batch_size=64):
        return DataLoader(
                        self.get_train_dataset(),
                        batch_size=batch_size,
                        collate_fn=default_data_collator
                    )

    def get_eval_dataloader(self,batch_size=64):
        return DataLoader(
                        self.get_eval_dataset(),
                        batch_size=batch_size,
                        collate_fn=default_data_collator
                    )
                            
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

def verify_backend(backend,
                  limit_train_frac,
                  data_location,
                  datareader_ray_obj_ref,
                  activity_level):
    
    if backend == "petastorm":
        if activity_level == "day":
            raise NotImplementedError("Day-level data is not yet supported with petastorm")
        
        if limit_train_frac:
            raise NotImplementedError("Petastorm relies on pre-processed data, so limit_train_frac can't be used yet")
        
        if data_location:
            raise NotImplementedError("With Petastorm please use --train_path and --eval_path")
        
        if datareader_ray_obj_ref:
            raise NotImplementedError("Petastorm backend does not support ray references")


class ActivityTask(Task):
    """ Is inhereited by anything that operatres over the minute
        level data
        
        TODO: A ton of these arguments only work with the legacy dask backend.
        Likely we want to remove them.
        """
    def __init__(self,base_dataset,dataset_args={},
                     activity_level = "minute",
                     look_for_cached_datareader=False,
                     datareader_ray_obj_ref=None,
                     cache=True,
                     only_with_lab_results=False,
                     limit_train_frac=None,
                     train_path=None,
                     eval_path=None,
                     test_path=None,
                     downsample_negative_frac=None,
                     shape=None,
                     keys=None,
                     normalize_numerical=True,
                     backend="petastorm"):
        
        super(ActivityTask,self).__init__()
        
        self.backend = backend
        if keys:
            self.keys = keys
            
        split_date = dataset_args.pop("split_date",None)
        eval_frac = dataset_args.pop("eval_frac",None)

        if not split_date and not eval_frac:
            raise KeyError("Must provide some strategy for splitting train\
                           and test. Either 'split_date' or 'eval_frac'")
        
        min_date = dataset_args.get("min_date",None)
        max_date = dataset_args.get("max_date",None)

        day_window_size = dataset_args.get("day_window_size",None)
        max_missing_days_in_window = dataset_args.get("max_missing_days_in_window",None)
        data_location = dataset_args.get("data_location",None)
        lab_results_reader = td.LabResultsReader()
        
        verify_backend(backend = backend,
                       limit_train_frac = limit_train_frac,
                       data_location = data_location,
                       datareader_ray_obj_ref = datareader_ray_obj_ref,
                       activity_level = activity_level)

        #### Original backend loads data on the fly with Dask ####
        if self.backend == 'dask':
            if only_with_lab_results:
                participant_ids = lab_results_reader.participant_ids
            else: 
                participant_ids = None

            if activity_level == "minute":
                base_activity_reader = td.MinuteLevelActivityReader
            else:
                base_activity_reader = td.DayLevelActivityReader

            if dataset_args.get("is_cached") and look_for_cached_datareader:
                activity_reader = load_cached_activity_reader(self.get_name(),
                                                            dataset_args=dataset_args,
                                                            fail_if_mismatched=True)
            elif datareader_ray_obj_ref:
                activity_reader = ray.get(datareader_ray_obj_ref)
            else:
                add_features_path = dataset_args.pop("add_features_path",None)
                activity_reader = base_activity_reader(min_date = min_date,
                                                    participant_ids=participant_ids,
                                                    max_date = max_date,
                                                    day_window_size=day_window_size,
                                                    max_missing_days_in_window=max_missing_days_in_window,
                                                    add_features_path=add_features_path,
                                                    data_location = data_location)

            if limit_train_frac:
                train_participant_dates, eval_participant_dates = activity_reader.split_participant_dates(date=split_date,eval_frac=eval_frac,
                                                                                                        limit_train_frac=limit_train_frac,
                                                                                                        by_participant=True)

            else:
                train_participant_dates, eval_participant_dates = activity_reader.split_participant_dates(date=split_date,eval_frac=eval_frac)
            

            self.train_dataset = base_dataset(activity_reader, lab_results_reader,
                            participant_dates = train_participant_dates,
                            cache=cache,**dataset_args)
            self.eval_dataset = base_dataset(activity_reader, lab_results_reader,
                                participant_dates = eval_participant_dates,
                                cache=cache,
                                **dataset_args)


        ### Newer backend relies on petastorm and is faster, but requires more pre-processing:
        elif self.backend == "petastorm":
            """
            Set the necessary attributes and adjust the time window of data  
            """
            #TODO make sure labler gets label for right day
            #TODO ensure labler is serialized properly 
            
            self.train_path = train_path
            self.eval_path = eval_path
            self.test_path = test_path

            self.train_url= url_from_path(train_path)
            self.eval_url = url_from_path(eval_path)
            self.test_url = url_from_path(test_path)
            
            labler = self.get_labler()

            if downsample_negative_frac:
                if not hasattr(labler,"get_positive_keys"):
                    raise ValueError(f"Tried to downsample negatives but {type(labler)}"
                                      " does not support `get_positive_keys`")
                positive_keys = labler.get_positive_keys()
                has_positive_predicate = peta_pred.in_lambda(["participant_id","end"],
                                                             lambda x,y : (x,pd.to_datetime(y)) in positive_keys)
                in_subset_predicate = peta_pred.in_pseudorandom_split([downsample_negative_frac,1-downsample_negative_frac],
                                                                      0,"id")
                self.predicate = peta_pred.in_reduce([has_positive_predicate, in_subset_predicate], any)
            else:
                self.predicate = None
                                                            
                
            infer_schema_path = None
            for path in [self.train_path,self.eval_path,self.test_path]:
                if path: 
                    infer_schema_path = path
                    break

            if not infer_schema_path:
                raise ValueError("Must provide at least one of {}"
                                "train_path, eval_path, or test_path"
                                "to use the petatstorm backend")
        
            schema = infer_or_load_unischema(ParquetDataset(infer_schema_path,validate_schema=False))
            fields = [k for k in schema.fields.keys() if not k in ["participant_id","id"]]
            # features = [k for k in schema.fields.keys() if not k in ["start","end","participant_id"]]
            
            def _transform_row(row):
                labler = self.get_labler()
                start = pd.to_datetime(row.pop("start"))
                #Because spark windows have and exclusive right boundary:
                end = pd.to_datetime(row.pop("end")) - pd.to_timedelta("1ms")

                participant_id = row.pop("participant_id")
                data_id = row.pop("id")

                if hasattr(self,"keys"):
                    keys = self.keys
                else:
                    keys = sorted(row.keys())

                results = []
                for k in keys:
                    feature_vector = row[k]
                    is_numerical = np.issubdtype(feature_vector.dtype, np.number)
                    
                    if normalize_numerical and is_numerical:
                        mu = feature_vector.mean()
                        sigma = feature_vector.std()
                        if sigma != 0:
                            feature_vector = (feature_vector - mu) / sigma
                    
                    results.append(feature_vector.T)

                label = int(labler(participant_id,start,end))
                return {"inputs_embeds": np.vstack(results).T,
                        "label": label,
                        "id": data_id,
                        "participant_id": participant_id,
                        "end_date_str": str(end)}
            
            new_fields = [("inputs_embeds",np.float32,None,False),
                        ("label",np.int_,None,False),
                        ("participant_id",np.str_,None,False),
                        ("id",np.int32,None,False),
                        ("end_date_str",np.str_,None,False)]

            self.transform = TransformSpec(_transform_row,removed_fields=fields,
                                                    edit_fields= new_fields)
            
            # Infer the shape of the data
            lengths = set()
            for k in self.keys:
                lengths.add(getattr(schema,k).shape[-1])
            lengths = set(lengths)
            if len(lengths) != 1:
                raise ValueError("Provided fields have mismatched feature sizes")
            else: 
                data_length = list(lengths)[0]
            
            self.data_shape = (data_length,len(self.keys))
        
        elif backend == "dynamic":
            self.data_shape = shape 

    def get_description(self):
        return self.__doc__

    def get_train_dataset(self):
        """
        In case the Petastorm framework is chosen, Returns a `Reader` instance
        """
        if self.backend in ["dask","dynamic"]:
            return self.train_dataset
        elif self.backend == "petastorm":
            logger.info("Making train dataset reader")
            return make_reader(self.train_url,transform_spec=self.transform,num_epochs=None)
        else:
            raise ValueError("Invalid backend")


    def get_eval_dataset(self):
        if self.backend in ["dask","dynamic"]:
            return self.eval_dataset
        elif self.backend == "petastorm":
            logger.info("Making eval dataset reader")
            return make_reader(self.eval_url,transform_spec=self.transform)
        else:
            raise ValueError("Invalid backend")
    
    
################################################
########### TASKS IMPLEMENTATIONS ##############
################################################


class DummySquareOrSine(ActivityTask, ClassificationMixin):
    """A dummy task to predict whether or not the total number of steps
       on the first day of a window is >= the mean across the whole dataset"""
    
    def __init__(self,dataset_args={}, n = 10000, p = 0.1, **kwargs):
        self.train_dataset = td.DummySquareOrSineHRDataset(n=n,p=p)
        self.eval_dataset = td.DummySquareOrSineHRDataset(n=n,p=p)
        self.data_shape = (4*24*60,1)
        ActivityTask.__init__(self, td.ActivtyDataset, 
                              shape = self.data_shape,
                              dataset_args=dataset_args,
                               **kwargs)
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

class GeqMeanSteps(ActivityTask, ClassificationMixin):
    """A dummy task to predict whether or not the total number of steps
       on the first day of a window is >= the mean across the whole dataset"""
    
    def __init__(self,dataset_args={}, **kwargs):
        ActivityTask.__init__(self,td.ActivtyDataset,dataset_args=dataset_args, **kwargs)
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
    """Predict whether a participant was positive
       given a rolling window of minute level activity data.
       We validate on data after split_date, but before
       max_date, if provided"""

    def __init__(self,dataset_args={}, activity_level = "minute",
                window_onset_max = 0, window_onset_min = 0,
                **kwargs):
        self.labler = FluPosLabler(window_onset_max=window_onset_max,
                                   window_onset_min=window_onset_min)

        dataset_args["labeler"] = self.labler
        self.keys = ['heart_rate',
                     'missing_heart_rate',
                     'missing_steps',
                     'sleep_classic_0',
                     'sleep_classic_1',
                     'sleep_classic_2',
                     'sleep_classic_3', 
                     'steps']

        ActivityTask.__init__(self,td.CustomLabler,dataset_args=dataset_args,
                                 activity_level=activity_level,**kwargs)
        ClassificationMixin.__init__(self)
        

    def get_name(self):
        return "PredictFluPos"
    
    def get_labler(self):
        return self.labler


class PredictWeekend(ActivityTask, ClassificationMixin):
    """Predict the whether the associated data belongs to a 
       weekend"""

    def __init__(self,dataset_args={}, activity_level = "minute", keys=None,
                **kwargs):
        self.labler = DayOfWeekLabler([5,6])
        if keys:
            self.keys=keys
        else:
            self.keys = ['heart_rate',
                        'missing_heart_rate',
                        'missing_steps',
                        'sleep_classic_0',
                        'sleep_classic_1',
                        'sleep_classic_2',
                        'sleep_classic_3', 
                        'steps']

        dataset_args["labeler"] = self.labler

        ActivityTask.__init__(self,td.CustomLabler,dataset_args=dataset_args,
                                 activity_level=activity_level,**kwargs)
        ClassificationMixin.__init__(self)
        
    def get_name(self):
        return "PredictWeekend"
    
    def get_labler(self):
        return self.labler

class PredictEvidationILI(ActivityTask, ClassificationMixin):
    """Predict the whether a participant was positive
       given a rolling window of minute level activity data.
       We validate on data after split_date, but before
       max_date, if provided"""

    def __init__(self, feather_path,
                       dataset_args={}, 
                       activity_level = "minute",
                       ili_types=[1,2,3],
                       window_onset_min = -5,
                       window_onset_max = 1,
                       **kwargs):
        
        #TODO would be nice to have a task_args field in the task spec yaml
        self.ili_types = ili_types
        self.keys =  ['heart_rate',
                     'missing_heart_rate',
                     'missing_steps',
                     'sleep_classic_0',
                     'sleep_classic_1',
                     'sleep_classic_2',
                     'sleep_classic_3', 
                     'steps']
        
        self.feather_path = feather_path,
        self.filename = os.path.basename(feather_path)
        self.window_onset_min = window_onset_min
        self.window_onset_max = window_onset_max

        self.labler = EvidationILILabler(feather_path,
                                         ili_types = self.ili_types,
                                         window_onset_min = self.window_onset_min,
                                         window_onset_max = self.window_onset_max)

        dataset_args["labeler"] = self.labler

        ActivityTask.__init__(self,td.CustomLabler,dataset_args=dataset_args,
                                 activity_level=activity_level,**kwargs)
        ClassificationMixin.__init__(self)
        

    def get_name(self):
        return f"PredictEvidationILI-types:{self.ili_types}-window_onset_min:{self.window_onset_min}"\
               f"-window_onset_max:{self.window_onset_max}-file:{self.filename}"
    
    def get_labler(self):
        return self.labler


class PredictTrigger(ActivityTask,ClassificationMixin):
    """Predict the whether a participant triggered the 
       test on the last day of a range of data"""

    def __init__(self,dataset_args={}, activity_level="minute", **kwargs):
        ActivityTask.__init__(self,td.PredictTriggerDataset,dataset_args=dataset_args,
                               activity_level = activity_level, **kwargs)
        ClassificationMixin.__init__(self)
        # self.is_classification = True

    def get_name(self):
        return "PredictTrigger"

class PredictSurveyClause(ActivityTask,ClassificationMixin):
    """Predict the whether a clause in the onehot
       encoded surveys is true for a given day. 
       
       For a sense of what kind of logical clauses are
       supported, check out:
    
       https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html"""

    def __init__(self,dataset_args={},activity_level="minute", **kwargs):
        self.clause = dataset_args.pop("clause")
        self.survey_responses = load_processed_table("daily_surveys_onehot").set_index("participant_id")
        self.labler = ClauseLabler(self.survey_responses,self.clause)
        dataset_args["labeler"] = self.labler
        ActivityTask.__init__(self, td.CustomLabler, dataset_args=dataset_args,
                             activity_level=activity_level, **kwargs)
        ClassificationMixin.__init__(self)
    
    def get_labler(self):
        return self.labler

    def get_name(self):
        return f"PredictSurveyCol-{self.clause}"

    def get_description(self):
        return self.__doc__


class SingleWindowActivityTask(Task):
    """Base class for tasks that make predictions about single windows of data
       (e.g.) predicting a user's BMI."""
    
    def __init__(self,base_dataset,dataset_args={}, activity_level="minute",
                window_selection="first",look_for_cached_datareader=False,
                datareader_ray_obj_ref=None, **kwargs):
        Task.__init__(self)
        eval_frac = dataset_args.pop("eval_frac",None)
        split_date = dataset_args.pop("split_date",None)

        min_date = dataset_args.get("min_date",None)
        max_date = dataset_args.get("max_date",None)
        day_window_size = dataset_args.get("day_window_size",None)
        max_missing_days_in_window = dataset_args.get("max_missing_days_in_window",None)

        lab_results_reader = td.LabResultsReader()
        participant_ids = lab_results_reader.participant_ids
        data_location = dataset_args.pop("data_location",None)

        if activity_level == "minute":
            base_activity_reader = td.MinuteLevelActivityReader
        else:
            base_activity_reader = td.DayLevelActivityReader

        if dataset_args.get("is_cached") and look_for_cached_datareader:
            activity_reader = load_cached_activity_reader(self.get_name(),
                                                          dataset_args=dataset_args,
                                                          fail_if_mismatched=True)
        elif datareader_ray_obj_ref:
            raise NotImplementedError
        else:
            add_features_path = dataset_args.pop("add_features_path",None)
            activity_reader = base_activity_reader(participant_ids=participant_ids,
                                                           min_date = min_date,
                                                           max_date = max_date,
                                                           day_window_size=day_window_size,
                                                           max_missing_days_in_window=max_missing_days_in_window,
                                                           add_features_path=add_features_path,
                                                           data_location=data_location)

        train_participant_dates, eval_participant_dates = activity_reader.split_participant_dates(date=split_date,eval_frac=eval_frac)
        
        if window_selection == "first":
            train_participant_dates = self.filter_participant_dates(train_participant_dates)
            eval_participant_dates = self.filter_participant_dates(eval_participant_dates)
    
        else:
            raise NotImplementedError

        limit_train_frac = dataset_args.get("limit_train_frac",None)
        if limit_train_frac:
            train_participant_dates = train_participant_dates[:int(len(train_participant_dates)*limit_train_frac)]

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

class ClassifyObese(ActivityTask, ClassificationMixin):
    def __init__(self, dataset_args, activity_level,**kwargs):
        self.labler = AudereObeseLabler()
        self.keys = ['heart_rate',
                'missing_heart_rate',
                'missing_steps',
                'sleep_classic_0',
                'sleep_classic_1',
                'sleep_classic_2',
                'sleep_classic_3', 
                'steps']
        dataset_args["labeler"] = self.labler
        ActivityTask.__init__(self, td.CustomLabler, 
                         dataset_args=dataset_args, 
                         activity_level=activity_level, 
                         **kwargs)
        ClassificationMixin.__init__(self)      

    def get_labler(self):
        return self.labler
    
    def get_name(self):
        return "ClassifyObese"

class EarlyDetection(ActivityTask):
    """Mimics the task used by Evidation Health"""

    def __init__(self,base_dataset=td.EarlyDetectionDataset,dataset_args={},
                      activity_level = "minute", look_for_cached_datareader=False):
        eval_frac = dataset_args.pop("eval_frac",None)

        if not eval_frac:
            raise KeyError("Must provide an eval fraction for splitting train and test")
        
        min_date = dataset_args.pop("min_date",None)
        max_date = dataset_args.pop("max_date",None)
        day_window_size = dataset_args.pop("day_window_size",4)
        window_pad = dataset_args.pop("window_pad",20)
        data_location = dataset_args.pop("data_location",None)
        max_missing_days_in_window = dataset_args.pop("max_missing_days_in_window",None)

        lab_results_reader = td.LabResultsReader(pos_only=True)
        participant_ids = lab_results_reader.participant_ids
        

        if activity_level == "minute":
            base_activity_reader = td.MinuteLevelActivityReader
        else:
            base_activity_reader = td.DayLevelActivityReader
            
        if dataset_args.get("is_cached") and look_for_cached_datareader:
            logger.info("Loading cached data reader...")
            activity_reader = load_cached_activity_reader(self.get_name())
        else:
            add_features_path = dataset_args.pop("add_features_path",None)
            activity_reader = base_activity_reader(participant_ids=participant_ids,
                                                           min_date = min_date,
                                                           max_date = max_date,
                                                           day_window_size=day_window_size,
                                                           max_missing_days_in_window=max_missing_days_in_window,
                                                           add_features_path=add_features_path,
                                                           data_location=data_location)


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

        limit_train_frac = dataset_args.get("limit_train_frac",None)
        if limit_train_frac:
            train_participant_dates = train_participant_dates[:int(len(train_participant_dates)*limit_train_frac)]

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
    