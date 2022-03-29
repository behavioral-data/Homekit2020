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

from importlib.resources import path
import sys
import os
from typing import Any, List, Optional, Tuple

from pyarrow.parquet import ParquetDataset
from sklearn.utils import resample


import numpy as np

import pytorch_lightning as pl

from petastorm import make_reader
from petastorm.transform import TransformSpec
from petastorm.etl.dataset_metadata import infer_or_load_unischema
import petastorm.predicates  as peta_pred
from petastorm.pytorch import DataLoader as PetastormDataLoader


from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY

from src.models.eval import classification_eval, regression_eval
from src.data.utils import load_processed_table, url_from_path
from src.utils import get_logger, read_yaml
from src.models.lablers import (FluPosLabler, ClauseLabler, EvidationILILabler, 
                                 DayOfWeekLabler, AudereObeseLabler, DailyFeaturesLabler,
                                 CovidLabler, SameParticipantLabler, SequentialLabler)

logger = get_logger(__name__)

import pandas as pd

DEFAULT_FIELDS = ['heart_rate',
                     'missing_heart_rate',
                     'missing_steps',
                     'sleep_classic_0',
                     'sleep_classic_1',
                     'sleep_classic_2',
                     'sleep_classic_3', 
                     'steps']


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
        super().__init__()
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
    # """Base class for tasks in this project"""
    # def __init__(self,train_path: Optional[str] = None):
    #     self.train_path = train_path
    
    def __init__(self,fields: Optional[List[str]] = None,
                     train_path: Optional[str] = None,
                     val_path: Optional[str] = None,
                     test_path: Optional[str] = None,
                     downsample_negative_frac: Optional[float] = None,
                     shape: Optional[Tuple[int, ...]] = None,
                     normalize_numerical: bool = True,
                     append_daily_features: bool = False,
                     daily_features_path: Optional[str] = None,
                     backend: str = "petastorm",
                     batch_size: int = 800,
                     activity_level: str = "minute"):

        #TODO does not currently support day level data   
        super(ActivityTask,self).__init__()
        self.fields = fields
        self.batch_size=batch_size
        self.backend = backend
        
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
                    keys = self.fields
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
            for k in self.fields:
                lengths.add(getattr(schema,k).shape[-1])
            lengths = set(lengths)
            if len(lengths) != 1:
                raise ValueError("Provided fields have mismatched feature sizes")
            else: 
                data_length = list(lengths)[0]
            
            self.data_shape = (data_length,len(self.fields))
        
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

@DATAMODULE_REGISTRY
class PredictDailyFeatures(ActivityTask, RegressionMixin):
    """Predict whether a participant was positive
       given a rolling window of minute level activity data.
       We validate on data after split_date, but before
       max_date, if provided"""

    def __init__(self, fields: List[str] = DEFAULT_FIELDS, 
                       activity_level: str = "minute",
                       window_size: int =7,
                **kwargs):
        self.labler = DailyFeaturesLabler(window_size=window_size)
        self.fields = ['heart_rate',
                     'missing_heart_rate',
                     'missing_steps',
                     'sleep_classic_0',
                     'sleep_classic_1',
                     'sleep_classic_2',
                     'sleep_classic_3', 
                     'steps']

        ActivityTask.__init__(self, fields=fields, activity_level=activity_level,**kwargs)
        RegressionMixin.__init__(self)
        

    def get_name(self):
        return "PredictDailyFeatures"
    
    def get_labler(self):
        return self.labler
 
        
@DATAMODULE_REGISTRY
class PredictFluPos(ActivityTask):
    """Predict whether a participant was positive
       given a rolling window of minute level activity data.
       We validate on data after split_date, but before
       max_date, if provided"""
    is_classification = True
    def __init__(self, fields: List[str] = DEFAULT_FIELDS, activity_level: str = "minute",
                window_onset_max: int = 0, window_onset_min:int = 0,
                **kwargs):
        
        self.is_classification = True
        self.labler = FluPosLabler(window_onset_max=window_onset_max,
                                   window_onset_min=window_onset_min)

        ActivityTask.__init__(self, fields=fields, activity_level=activity_level,**kwargs)
        # ClassificationMixin.__init__(self)
        

    def get_name(self):
        return "PredictFluPos"
    
    def get_labler(self):
        return self.labler

@DATAMODULE_REGISTRY
class PredictWeekend(ActivityTask, ClassificationMixin):
    """Predict the whether the associated data belongs to a 
       weekend"""

    def __init__(self, fields: List[str] = DEFAULT_FIELDS, 
                       activity_level: str = "minute",
                       **kwargs):

        self.labler = DayOfWeekLabler([5,6])
        ActivityTask.__init__(self, fields=fields, activity_level=activity_level,**kwargs)
        ClassificationMixin.__init__(self)
        
    def get_name(self):
        return "PredictWeekend"
    
    def get_labler(self):
        return self.labler


@DATAMODULE_REGISTRY
class PredictCovidSmall(ActivityTask, ClassificationMixin):
    """Predict the whether a participant was diagnosed with
    covid on the final day of the window
    
    This was designed for data from Mirsha et. al,
    and uses the processed results from 
    /projects/bdata/datasets/covid-fitbit/processed/covid_dates.csv
    """

    def __init__(self, dates_path: str,
                       fields: List[str] = DEFAULT_FIELDS, 
                       activity_level: str = "minute",
                       **kwargs):
        
        self.dates_path = dates_path
        self.filename = os.path.basename(dates_path)

        self.labler = CovidLabler(dates_path)
        ActivityTask.__init__(self, fields=fields, activity_level=activity_level,**kwargs)
        ClassificationMixin.__init__(self)
        

    def get_name(self):
        return "PredictCovidSmall"
    
    def get_labler(self):
        return self.labler

@DATAMODULE_REGISTRY
class PredictSurveyClause(ActivityTask,ClassificationMixin):
    """Predict the whether a clause in the onehot
       encoded surveys is true for a given day. 
       
       For a sense of what kind of logical clauses are
       supported, check out:
    
       https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html"""

    def __init__(self, clause: str, 
                       activity_level: str = "minute", 
                       fields: List[str] = DEFAULT_FIELDS, 
                       survey_path: Optional[str] = None,
                       **kwargs):
        self.clause = clause
        self.survey_responses = load_processed_table("daily_surveys_onehot",path=survey_path).set_index("participant_id")
        self.labler = ClauseLabler(self.survey_responses,self.clause)
        ActivityTask.__init__(self, fields=fields, activity_level=activity_level,**kwargs)
        ClassificationMixin.__init__(self)
    
    def get_labler(self):
        return self.labler

    def get_name(self):
        return f"PredictSurveyCol-{self.clause}"

    def get_description(self):
        return self.__doc__

@DATAMODULE_REGISTRY
class ClassifyObese(ActivityTask, ClassificationMixin):
    def __init__(self, activity_level: str = "minute", 
                       fields: List[str] = DEFAULT_FIELDS,
                       **kwargs):

        self.labler = AudereObeseLabler()    
        ActivityTask.__init__(self, fields=fields, activity_level=activity_level,**kwargs)
        ClassificationMixin.__init__(self)      

    def get_labler(self):
        return self.labler
    
    def get_name(self):
        return "ClassifyObese"