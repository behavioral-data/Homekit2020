
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import List

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models.tasks import ActivityTask


"""
A RowTransform is a callable object that takes a row in 
a petastorm dataset and returns an entry that is in turn
composed into a batch.
"""


class RowTransform(ABC):
    def __init__(self, task : "ActivityTask", 
                       **kwargs):

        self.task = task

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_new_fields(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_removed_fields(self):
        pass

class DefaultTransformRow(RowTransform):
    def __init__(self, *args,
                       normalize_numerical : bool = False,
                       **kwargs):

        self.normalize_numerical = normalize_numerical
        super().__init__(*args, **kwargs)

    def __call__(self, row):
        return self._transform_row(row)
    
    def get_new_fields(self, *args, **kwargs):

        fields = [("inputs_embeds",np.float32,None,False),
                        ("label",np.int_,None,False),
                        ("participant_id",np.str_,None,False),
                        ("id",np.int32,None,False),
                        ("end_date_str",np.str_,None,False)]


        for meta_key, meta_type in zip(self.task.get_metadata_lablers().keys(), self.task.get_metadata_types()):
            fields.append((meta_key,meta_type,None,False))

        return fields

    
    def get_removed_fields(self):
        return [k for k in self.task.schema.fields.keys() if not k in ["participant_id","id"]]

    def _transform_row(self,row):
        labler = self.task.get_labler()
        metadata_lablers = self.task.get_metadata_lablers()

        start = pd.to_datetime(row.pop("start"))
        #Because spark windows have and exclusive right boundary:
        end = pd.to_datetime(row.pop("end")) - pd.to_timedelta("1ms")

        participant_id = row.pop("participant_id")
        data_id = row.pop("id")

        if hasattr(self.task,"keys"):
            keys = self.task.keys
        elif self.task.fields:
            keys = self.task.fields 
        else:
            keys = sorted(row.keys())

        results = []
        for k in keys:
            feature_vector = row[k]
            is_numerical = np.issubdtype(feature_vector.dtype, np.number)
            
            if self.normalize_numerical and is_numerical:
                mu = feature_vector.mean()
                sigma = feature_vector.std()
                if sigma != 0:
                    feature_vector = (feature_vector - mu) / sigma
            
            results.append(feature_vector.T)
        inputs_embeds = np.vstack(results).T
        if not self.task.is_autoencoder:
            label = labler(participant_id,start,end)
            
        else:
            label = inputs_embeds.astype(np.float32)

        if self.task.daily_features_labler:
            day_features = self.task.daily_features_labler(participant_id,start,end)
            label = np.concatenate([[label],day_features])
            
        transform = {"inputs_embeds": inputs_embeds,
                    "label": label,
                    "id": data_id,
                    "participant_id": participant_id,
                    "end_date_str": str(end)}

        # appends metadata to batch
        for k, l in metadata_lablers.items():
            transform[k] = l(participant_id,start,end)

        return transform


class ContrastiveTransformRow(RowTransform):
    def __init__(self, *args,
                       normalize_numerical : bool = False,
                       **kwargs):

        self.normalize_numerical = normalize_numerical
        super().__init__(*args, **kwargs)

    def __call__(self, row):
        return self._transform_row(row)
    
    def get_new_fields(self, *args, **kwargs):

        fields = [("inputs_embeds",np.float32,None,False),
                        ("label",np.int_,None,False),
                        ("participant_id",np.str_,None,False),
                        ("id",np.int32,None,False),
                        ("end_date_str",np.str_,None,False)]


        for meta_key, meta_type in zip(self.task.get_metadata_lablers().keys(), self.task.get_metadata_types()):
            fields.append((meta_key,meta_type,None,False))

        return fields

    
    def get_removed_fields(self):
        return [k for k in self.task.schema.fields.keys() if not k in ["participant_id","id"]]

    def _transform_row(self,row):
        labler = self.task.get_labler()
        metadata_lablers = self.task.get_metadata_lablers()

        start_l = pd.to_datetime(row.pop("start_l"))
        start_r = pd.to_datetime(row.pop("start_l"))
        #Because spark windows have and exclusive right boundary:
        end_l = pd.to_datetime(row.pop("end_l")) - pd.to_timedelta("1ms")
        end_r = pd.to_datetime(row.pop("end_r")) - pd.to_timedelta("1ms")

        participant_id_l = row.pop("participant_id_l")
        participant_id_r = row.pop("participant_id_r")

        data_id = row.pop("id")

        if hasattr(self.task,"keys"):
            keys = self.task.keys
        elif self.task.fields:
            keys = self.task.fields 
        else:
            keys = sorted(row.keys())

        results_l = []
        results_r = []

        for k in keys:
            feature_vector = row[k]
            is_numerical = np.issubdtype(feature_vector.dtype, np.number)
            
            if self.normalize_numerical and is_numerical:
                mu = feature_vector.mean()
                sigma = feature_vector.std()
                if sigma != 0:
                    feature_vector = (feature_vector - mu) / sigma
            
            if k.endswith("_l"):
                results_l.append(feature_vector.T)
            else:
                results_r.append(feature_vector.T)
    
        inputs_embeds_l = np.vstack(results_l).T
        inputs_embeds_r = np.vstack(results_l).T

        label = labler(participant_id_l,start_l,end_l,
                       participant_id_r,start_r,end_r)

        transform = {"inputs_embeds_l": inputs_embeds_l,
                    "inputs_embeds_r": inputs_embeds_r,
                    "label": label,
                    "id": data_id,
                    "participant_id_l": participant_id_l,
                    "participant_id_r": participant_id_r,
                    "end_date_str_l": str(end_l),
                    "end_date_str_r": str(end_r)}
        
        return transform