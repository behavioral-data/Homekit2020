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
from unicodedata import normalize
from warnings import WarningMessage
import os

from pyarrow.parquet import ParquetDataset
from sklearn.utils import resample

from torch.utils.data import DataLoader
from transformers.data.data_collator import default_data_collator
import ray

import numpy as np

import pytorch_lightning as pl

from petastorm import make_reader
from petastorm.transform import TransformSpec
from petastorm.etl.dataset_metadata import infer_or_load_unischema
import petastorm.predicates  as peta_pred
from petastorm.pytorch import DataLoader as PetastormDataLoader

from src.models.eval import classification_eval, regression_eval
from src.data.utils import load_processed_table, load_cached_activity_reader, url_from_path
from src.utils import get_logger, read_yaml
from src.models.lablers import (FluPosLabler, ClauseLabler, EvidationILILabler, 
                                 DayOfWeekLabler, AudereObeseLabler, DailyFeaturesLabler,
                                 CovidLabler, SameParticipantLabler, SequentialLabler)

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

def stack_keys(keys,row, normalize_numerical=True):
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
        
    return np.vstack(results).T

class Task(pl.LightningDataModule):
    def __init__(self):
        super().__init__(self)
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

    def get_val_dataset(self):
        return self.eval_dataset
    
    def get_labler(self):
        return NotImplementedError
    
    def get_train_dataloader(self,batch_size=64):
        return DataLoader(
                        self.get_train_dataset(),
                        batch_size=batch_size,
                        collate_fn=default_data_collator
                    )

    def get_val_dataloader(self,batch_size=64):
        return DataLoader(
                        self.get_val_dataset(),
                        batch_size=batch_size,
                        collate_fn=default_data_collator
                    )

class TaskTypeMixin():
    def __init__(self):
        self.is_regression=False                            
        self.is_classification=False
        self.is_autoencoder=False
        self.is_double_encoding = False

class ClassificationMixin(TaskTypeMixin):
    def __init__(self):
        TaskTypeMixin.__init__(self)
        self.is_classification = True

    def evaluate_results(self,logits,labels,threshold=0.5):
        return classification_eval(logits,labels,threshold=threshold)

    def get_huggingface_metrics(self,threshold=0.5):
        def evaluator(pred):
            labels = pred.label_ids
            logits = pred.predictions
            return self.evaluate_results(logits,labels,threshold=threshold)
        return evaluator
class RegressionMixin(TaskTypeMixin):
    def __init__(self):
        TaskTypeMixin.__init__(self)
        self.is_regression=True
    
    def evaluate_results(self,preds,labels):
        return regression_eval(preds,labels)

    def get_huggingface_metrics(self):
        def evaluator(predictions):
            labels = predictions.label_ids
            preds = predictions.predictions
            return self.evaluate_results(preds,labels)
        return evaluator
class AutoencodeMixin(RegressionMixin):
    def __init__(self):
        self.is_autoencoder = True
        super(AutoencodeMixin,self).__init__()

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
            raise NotImplementedError("With Petastorm please use --train_path and --val_path")
        
        if datareader_ray_obj_ref:
            raise NotImplementedError("Petastorm backend does not support ray references")


class ActivityTask(Task):
    """Base class for tasks in this project"""
    def __init__(self,train_path=None,
                     val_path=None,
                     test_path=None,
                     downsample_negative_frac=None,
                     shape=None,
                     keys=None,
                     normalize_numerical=True,
                     append_daily_features=False,
                     daily_features_path=None,
                     double_encode=False,
                     backend="petastorm",
                     batch_size: int = 600,
                     activity_level="minute"):

        #TODO does not currently support day level data   
        super(ActivityTask,self).__init__()
        
        self.batch_size=batch_size
        self.backend = backend
        if keys:
            self.keys = keys
        
        self.daily_features_appended = append_daily_features
        if self.daily_features_appended:
            self.daily_features_labler = DailyFeaturesLabler(data_location=daily_features_path, window_size=1)
        else:
            self.daily_features_labler = None

        ### Newer backend relies on petastorm and is faster, but requires more pre-processing:
        if self.backend == "petastorm":
            """
            Set the necessary attributes and adjust the time window of data  
            """
            #TODO make sure labler gets label for right day
            #TODO ensure labler is serialized properly 
            
            self.train_path = train_path
            self.val_path = val_path
            self.test_path = test_path

            self.train_url= url_from_path(train_path)
            self.val_url = url_from_path(val_path)
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
            for path in [self.train_path,self.val_path,self.test_path]:
                if path: 
                    infer_schema_path = path
                    break

            if not infer_schema_path:
                raise ValueError("Must provide at least one of "
                                "train_path, val_path, or test_path")
        
            
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
                inputs_embeds = np.vstack(results).T
                if not self.is_autoencoder:
                    label = labler(participant_id,start,end)
                    
                else:
                    label = inputs_embeds.astype(np.float32)

                if self.daily_features_labler:
                    day_features = self.daily_features_labler(participant_id,start,end)
                    label = np.concatenate([[label],day_features])
                    

                return {"inputs_embeds": inputs_embeds,
                        "label": label,
                        "id": data_id,
                        "participant_id": participant_id,
                        "end_date_str": str(end)}
        
            if not self.is_autoencoder:
                label_type = np.int_
            else:
                label_type = np.float32

            new_fields = [("inputs_embeds",np.float32,None,False),
                        ("label",label_type,None,False),
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
            
            if double_encode:
                self.data_shape = (data_length,len(self.keys)//2)
            else:
                self.data_shape = (data_length,len(self.keys))
        
        elif backend == "dynamic":
            self.data_shape = shape 
        
        self.save_hyperparameters()

    def get_description(self):
        return self.__doc__

    def train_dataloader(self):
        if self.train_url:
            return PetastormDataLoader(make_reader(self.train_url,transform_spec=self.transform,
                                                    predicate=self.predicate),
                                    batch_size=self.batch_size)        

    def val_dataloader(self):
        if self.val_url:
            return PetastormDataLoader(make_reader(self.val_url,transform_spec=self.transform,
                                                    predicate=self.predicate),
                                        batch_size=self.batch_size)   
    def test_dataloader(self):
        if self.val_url:
            return PetastormDataLoader(make_reader(self.test_url,transform_spec=self.transform,
                                                    predicate=self.predicate),
                                        batch_size=self.batch_size)   

    def add_task_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Task")
        return parent_parser
    
################################################
########### TASKS IMPLEMENTATIONS ##############
################################################

class PredictDailyFeatures(ActivityTask, RegressionMixin):
    """Predict whether a participant was positive
       given a rolling window of minute level activity data.
       We validate on data after split_date, but before
       max_date, if provided"""

    def __init__(self, activity_level = "minute",
                window_size=7,
                **kwargs):
        self.labler = DailyFeaturesLabler(window_size=window_size)
        self.keys = ['heart_rate',
                     'missing_heart_rate',
                     'missing_steps',
                     'sleep_classic_0',
                     'sleep_classic_1',
                     'sleep_classic_2',
                     'sleep_classic_3', 
                     'steps']

        ActivityTask.__init__(self, activity_level=activity_level,**kwargs)
        RegressionMixin.__init__(self)
        

    def get_name(self):
        return "PredictDailyFeatures"
    
    def get_labler(self):
        return self.labler

class PredictFluPos(ActivityTask, ClassificationMixin):
    """Predict whether a participant was positive
       given a rolling window of minute level activity data.
       We validate on data after split_date, but before
       max_date, if provided"""

    def __init__(self, activity_level = "minute",
                window_onset_max = 0, window_onset_min = 0,
                **kwargs):
        self.labler = FluPosLabler(window_onset_max=window_onset_max,
                                   window_onset_min=window_onset_min)
        self.keys = ['heart_rate',
                     'missing_heart_rate',
                     'missing_steps',
                     'sleep_classic_0',
                     'sleep_classic_1',
                     'sleep_classic_2',
                     'sleep_classic_3', 
                     'steps']

        ActivityTask.__init__(self, activity_level=activity_level, **kwargs)
        ClassificationMixin.__init__(self)
        

    def get_name(self):
        return "PredictFluPos"
    
    def get_labler(self):
        return self.labler


class PredictWeekend(ActivityTask, ClassificationMixin):
    """Predict the whether the associated data belongs to a 
       weekend"""

    def __init__(self, activity_level = "minute", keys=None,
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


        ActivityTask.__init__(self, activity_level=activity_level,**kwargs)
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



        ActivityTask.__init__(self, activity_level=activity_level,**kwargs)
        ClassificationMixin.__init__(self)
        

    def get_name(self):
        return f"PredictEvidationILI-types:{self.ili_types}-window_onset_min:{self.window_onset_min}"\
               f"-window_onset_max:{self.window_onset_max}-file:{self.filename}"
    
    def get_labler(self):
        return self.labler



class PredictCovidSmall(ActivityTask, ClassificationMixin):
    """Predict the whether a participant was diagnosed with
    covid on the final day of the window
    
    This was designed for data from Mirsha et. al,
    and uses the processed results from 
    /projects/bdata/datasets/covid-fitbit/processed/covid_dates.csv
    """

    def __init__(self, dates_path,
                       activity_level = "minute",
                       **kwargs):
        
        self.keys =  ['heart_rate',
                     'missing_heart_rate',
                     'missing_steps',
                     'sleep_classic_0',
                     'sleep_classic_1',
                     'sleep_classic_2',
                     'sleep_classic_3', 
                     'steps']
        
        self.dates_path = dates_path,
        self.filename = os.path.basename(dates_path)

        self.labler = CovidLabler(dates_path)
        ActivityTask.__init__(self, activity_level=activity_level,**kwargs)
        ClassificationMixin.__init__(self)
        

    def get_name(self):
        return "PredictCovidSmall"
    
    def get_labler(self):
        return self.labler


class PredictTrigger(ActivityTask,ClassificationMixin):
    """Predict the whether a participant triggered the 
       test on the last day of a range of data"""

    def __init__(self, activity_level="minute", **kwargs):
        ActivityTask.__init__(self, activity_level = activity_level, **kwargs)
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

    def __init__(self,clause, activity_level="minute", **kwargs):
        self.clause = clause
        self.keys = ['heart_rate',
                'missing_heart_rate',
                'missing_steps',
                'sleep_classic_0',
                'sleep_classic_1',
                'sleep_classic_2',
                'sleep_classic_3', 
                'steps']
        self.survey_responses = load_processed_table("daily_surveys_onehot").set_index("participant_id")
        self.labler = ClauseLabler(self.survey_responses,self.clause)
        ActivityTask.__init__(self, activity_level=activity_level, **kwargs)
        ClassificationMixin.__init__(self)
    
    def get_labler(self):
        return self.labler

    def get_name(self):
        return f"PredictSurveyCol-{self.clause}"

    def get_description(self):
        return self.__doc__

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
        ActivityTask.__init__(self, activity_level=activity_level, 
                         **kwargs)
        ClassificationMixin.__init__(self)      

    def get_labler(self):
        return self.labler
    
    def get_name(self):
        return "ClassifyObese"

class Autoencode(AutoencodeMixin, ActivityTask):
    """Autoencode minute level data"""

    def __init__(self,dataset_args={},**kwargs):

        self.keys = ['heart_rate',
                'missing_heart_rate',
                'missing_steps',
                'sleep_classic_0',
                'sleep_classic_1',
                'sleep_classic_2',
                'sleep_classic_3', 
                'steps']
                     
        ActivityTask.__init__(self, **kwargs)
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
