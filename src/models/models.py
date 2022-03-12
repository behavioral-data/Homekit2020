"""
====================================================
Architectures For Behavioral Representation Learning     
====================================================
`Project repository available here  <https://github.com/behavioral-data/SeattleFluStudy>`_

This module contains the architectures used for behavioral representation learning in the reference paper. 
Particularly, the two main classes in the module implement a CNN architecture and the novel CNN-Transformer
architecture. 

**Classes**
    :class CNNEncoder: 
    :class CNNToTransformerEncoder:

"""

from copy import copy
from doctest import OutputChecker
from errno import ENXIO
from tokenize import Number

from typing import Dict,  Union, Any, Optional
import os
from random import sample

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import pandas as pd

from petastorm.pytorch import BatchedDataLoader
from petastorm.reader import Reader

from pytorch_lightning.loggers.wandb import WandbLogger
import torchmetrics

from src.models.losses import build_loss_fn
import src.models.modules as modules
from src.utils import get_logger, upload_pandas_df_to_wandb, binary_logits_to_pos_probs
from src.models.eval import (wandb_roc_curve, wandb_pr_curve, wandb_detection_error_tradeoff_curve,
                            classification_eval,  TorchMetricClassification, TorchMetricRegression, TorchMetricAutoencode)
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from wandb.plot.roc_curve import roc_curve

logger = get_logger(__name__)
class Config(object):
    def __init__(self,data) -> None:
        self.data = data
    def to_dict(self):
        return self.data


def get_config_from_locals(locals,model_type=None):
    locals = copy(locals)
    locals.pop("self",None)
    name = getattr(locals.pop("__class__",None),"__name__",None)
    locals["model_base_class"] = name

    model_specific_kwargs = locals.pop("model_specific_kwargs",{})
    locals.update(model_specific_kwargs)
    return Config(locals)


    
class SensingModel(pl.LightningModule):
    '''
    This is the base class for building sensing models.
    All trainable models should subclass this.
    '''

    def __init__(self, metric_class : torchmetrics.MetricCollection, 
                       bootstrap_val_metrics : bool = True,
                       learning_rate : Number = 1e-3,
                       warmup_steps : int = 0,
                       batch_size : int = 800,):
        
        super(SensingModel,self).__init__()
        self.val_preds = []
        self.train_labels = []
        
        self.val_preds = []
        self.val_labels =[]

        self.test_preds = []
        self.test_labels =[]
        self.test_participant_ids = []
        self.test_dates = []
        self.test_losses = []

        self.train_dataset = None
        self.eval_dataset=None

        self.train_metrics = metric_class(bootstrap_cis=False, prefix="train/")
        self.val_metrics = metric_class(bootstrap_cis=bootstrap_val_metrics, prefix="eval/")
        self.test_metrics = metric_class(bootstrap_cis=bootstrap_val_metrics, prefix="test/")

        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size

        self.wandb_id = None
        self.name = None

        self.save_hyperparameters()

    def on_train_start(self) -> None:
        self.train_metrics.apply(lambda x: x.to(self.device))
        self.val_metrics.apply(lambda x: x.to(self.device))
        return super().on_train_start()
        
    def training_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:

        x = batch["inputs_embeds"].type(torch.cuda.FloatTensor)
        y = batch["label"]

        loss,preds = self.forward(x,y)
        
        self.log("train/loss", loss.item(),on_step=True)
        preds = preds.detach()


        y = y.detach()
        self.train_metrics.update(preds,y)

        if self.is_classifier:
            self.val_preds.append(preds.detach().cpu())
            self.train_labels.append(y.detach().cpu())

        return {"loss":loss, "preds": preds, "labels":y}

    def on_train_epoch_end(self):
        
        # We get a DummyExperiment outside the main process (i.e. global_rank > 0)
        if os.environ.get("LOCAL_RANK","0") == "0" and self.is_classifier and isinstance(self.logger, WandbLogger):
            train_preds = torch.cat(self.val_preds, dim=0)
            train_labels = torch.cat(self.train_labels, dim=0)
            self.logger.experiment.log({"train/roc": wandb_roc_curve(train_preds,train_labels, limit = 9999)}, commit=False)
            self.logger.experiment.log({"train/pr": wandb_pr_curve(train_preds,train_labels)}, commit=False)
            self.logger.experiment.log({"train/det": wandb_detection_error_tradeoff_curve(train_preds,train_labels, limit=9999)}, commit=False)
        
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True)
        
        # Clean up for next epoch:
        self.train_metrics.reset()
        self.val_preds = []
        self.train_labels = []
        super().on_train_epoch_end()
    
    def on_train_epoch_start(self):
        self.train_metrics.to(self.device)
    
    def on_validation_epoch_start(self):
        torch.cuda.empty_cache()
        self.test_preds = []
    
    def on_test_epoch_end(self):
        # We get a DummyExperiment outside the main process (i.e. global_rank > 0)
        test_preds = torch.cat(self.test_preds, dim=0)
        test_labels = torch.cat(self.test_labels, dim=0)
        test_dates = np.concatenate(self.test_dates, axis=0)
        test_participant_ids = np.concatenate(self.test_participant_ids, axis=0)

        if os.environ.get("LOCAL_RANK","0") == "0" and self.is_classifier and isinstance(self.logger, WandbLogger):
            
            self.logger.experiment.log({"test/roc": wandb_roc_curve(test_preds,test_labels, limit = 9999)}, commit=False)
            self.logger.experiment.log({"test/pr": wandb_pr_curve(test_preds,test_labels)}, commit=False)
            self.logger.experiment.log({"test/det": wandb_detection_error_tradeoff_curve(test_preds,test_labels, limit=9999)}, commit=False)
        
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        
        #TODO This should probably be in its own method
        pos_probs = binary_logits_to_pos_probs(test_preds.cpu().numpy()) 
        self.predictions_df = pd.DataFrame(zip(test_participant_ids,test_dates,test_labels.cpu().numpy(),pos_probs),
                                        columns = ["participant_id","date","label","pred"])                                 

        # Clean up
        self.test_metrics.reset()
        self.test_preds = []
        self.test_labels = []
        self.test_participant_ids = []
        self.test_dates = []
        super().on_validation_epoch_end()
    
    def predict_step(self, batch: Any) -> Any:
        x = batch["inputs_embeds"].type(torch.cuda.FloatTensor)
        y = batch["label"]

        with torch.no_grad():
            loss,logits = self.forward(x,y)
            probs = torch.nn.functional.softmax(logits,dim=1)[:,-1]
        
        return {"loss":loss, "preds": logits, "labels":y,
                "participant_id":batch["participant_id"],
                "end_date":batch["end_date_str"] }

    def test_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:

        x = batch["inputs_embeds"].type(torch.cuda.FloatTensor)
        y = batch["label"]
        dates = batch["end_date_str"]
        participant_ids = batch["participant_id"]

        loss,preds = self.forward(x,y)

        self.log("test/loss", loss.item(),on_step=True,sync_dist=True)

        if self.multitask_daily_features:
            y = y[:,0].type(torch.int64)

        self.test_preds.append(preds.detach())
        self.test_labels.append(y.detach())
        self.test_participant_ids.append(participant_ids)
        self.test_dates.append(dates)

        self.test_metrics.update(preds,y)
        return {"loss":loss, "preds": preds, "labels":y}
        

    def on_validation_epoch_end(self):
        # We get a DummyExperiment outside the main process (i.e. global_rank > 0)
        if os.environ.get("LOCAL_RANK","0") == "0" and self.is_classifier and isinstance(self.logger, WandbLogger):
            val_preds = torch.cat(self.val_preds, dim=0)
            val_labels = torch.cat(self.val_labels, dim=0)
            self.logger.experiment.log({"eval/roc": wandb_roc_curve(val_preds,val_labels, limit = 9999)}, commit=False)
            self.logger.experiment.log({"eval/pr":  wandb_pr_curve(val_preds,val_labels)}, commit=False)
            self.logger.experiment.log({"eval/det": wandb_detection_error_tradeoff_curve(val_preds,val_labels, limit=9999)}, commit=False)
        
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        
        # Clean up
        self.val_metrics.reset()
        self.val_preds = []
        self.val_labels = []
        
        super().on_validation_epoch_end()

    def validation_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        x = batch["inputs_embeds"].type(torch.cuda.FloatTensor)
        y = batch["label"]
        
        loss,preds = self.forward(x,y)
        
        self.log("eval/loss", loss.item(),on_step=True,sync_dist=True)
        
        if self.multitask_daily_features:
            y = y[:,0].type(torch.int64)
            
        if self.is_classifier:
            self.val_preds.append(preds.detach())
            self.val_labels.append(y.detach())
        
        self.val_metrics.update(preds,y)
        return {"loss":loss, "preds": preds, "labels":y}
    
    def configure_optimizers(self):
        #TODO: Add support for other optimizers and lr schedules?
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        def scheduler(step):  
            return min(1., float(step + 1) / self.warmup_steps)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,scheduler)
        
        return [optimizer], [       
            {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1,
                'reduce_on_plateau': False,
                'monitor': 'val_loss',
            }
        ]

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False,
    ):
        optimizer.step(closure=optimizer_closure)

    
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    logger.info(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                logger.info(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def upload_predictions_to_wandb(self):
        upload_pandas_df_to_wandb(run_id=self.hparams.wandb_id,
                                  table_name="test_predictions",
                                  df=self.predictions_df)


class ModelTypeMixin():
    def __init__(self):
        self.is_regressor = False                            
        self.is_classifier = False
        self.is_autoencoder = False
        self.is_double_encoding = False

        self.metric_class = None

class ClassificationModel(SensingModel,ModelTypeMixin):
    '''
    Represents classification models 
    '''
    def __init__(self,**kwargs) -> None:
        SensingModel.__init__(self,TorchMetricClassification,**kwargs)
        self.is_classifier = True


class RegressionModel(SensingModel,ModelTypeMixin):
    def __init__(self,**kwargs) -> None:
        SensingModel.__init__(self,TorchMetricRegression,**kwargs)
        self.is_regressor = True
        
class CNNToTransformerClassifier(ClassificationModel):
    
    def __init__(self, input_features : int, num_attention_heads : int, num_hidden_layers: int, n_timesteps : int, kernel_sizes=[5,3,1], out_channels = [256,128,64], 
                stride_sizes=[2,2,2], dropout_rate=0.3, num_labels=2, learning_rate=1e-3, warmup_steps=100,
                skip_cnn=False, positional_encoding = False, model_head="classification", **kwargs) -> None:

        super().__init__(warmup_steps=warmup_steps, learning_rate=learning_rate)
        
        self.name = "CNNTransformerClassifier"
        self.criterion = nn.CrossEntropyLoss()
        self.encoder = modules.CNNToTransformerEncoder(input_features, num_attention_heads, num_hidden_layers,
                                                      n_timesteps, kernel_sizes=kernel_sizes, out_channels=out_channels,
                                                      stride_sizes=stride_sizes, dropout_rate=dropout_rate, num_labels=num_labels,
                                                      learning_rate=learning_rate, warmup_steps=warmup_steps, skip_cnn=skip_cnn,
                                                      positional_encoding=positional_encoding)
        
        self.head = modules.RegressionModule(self.encoder.d_model, self.encoder.final_length, num_labels)
        self.save_hyperparameters()
        
    def forward(self, inputs_embeds,labels):
        encoding = self.encoder.encode(inputs_embeds)
        preds = self.head(encoding)
        loss =  self.criterion(preds,labels)
        return loss, preds
    
        