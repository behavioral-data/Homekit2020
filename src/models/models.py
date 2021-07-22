from copy import copy
from typing import Union, Dict
from pytorch_lightning.core.step_result import Result

import gc

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import dropout
from torch.utils.data import DataLoader

import pytorch_lightning as pl

import numpy as np
import torchmetrics
from transformers import PretrainedConfig
import wandb
from wandb.plot.roc_curve import roc_curve
from wandb.data_types import Table


from src.SAnD.core import modules
from src.models.losses import build_loss_fn
from src.utils import check_for_wandb_run

class Config(object):
    def __init__(self,data) -> None:
        self.data = data
    def to_dict(self):
        return self.data


def conv_l_out(l_in,kernel_size,stride,padding=0, dilation=1):
    return np.floor((l_in + 2 * padding - dilation * (kernel_size-1)-1)/stride + 1)

def get_final_conv_l_out(l_in,kernel_sizes,stride_sizes,
                        max_pool_kernel_size=None, max_pool_stride_size=None):
    l_out = l_in
    for kernel_size, stride_size in zip(kernel_sizes,stride_sizes):
        l_out = conv_l_out(l_out,kernel_size,stride_size)
        if max_pool_kernel_size and max_pool_kernel_size:
            l_out = conv_l_out(l_out, max_pool_kernel_size,max_pool_stride_size)
    return int(l_out)

def get_config_from_locals(locals,model_type=None):
    locals = copy(locals)
    locals.pop("self",None)
    name = getattr(locals.pop("__class__",None),"__name__",None)
    locals["model_base_class"] = name

    model_specific_kwargs = locals.pop("model_specific_kwargs",{})
    locals.update(model_specific_kwargs)
    return Config(locals)

class CNNEncoder(nn.Module):
    def __init__(self, input_features, n_timesteps,
                kernel_sizes=[1], out_channels = [128], 
                stride_sizes=[1], max_pool_kernel_size = 3,
                max_pool_stride_size=2) -> None:

        n_layers = len(kernel_sizes)
        assert len(out_channels) == n_layers
        assert len(stride_sizes) == n_layers

        super(CNNEncoder,self).__init__()
        self.input_features = input_features

        layers = []
        for i in range(n_layers):
            if i == 0:
                in_channels = input_features
            else:
                in_channels = out_channels[i-1]
            layers.append(nn.Conv1d(in_channels = in_channels,
                                    out_channels = out_channels[i],
                                    kernel_size=kernel_sizes[i],
                                    stride = stride_sizes[i]))
            layers.append(nn.ReLU())
            if max_pool_stride_size and max_pool_kernel_size:
                layers.append(nn.MaxPool1d(max_pool_kernel_size, stride=max_pool_stride_size))
        self.layers = nn.ModuleList(layers)
        self.final_output_length = get_final_conv_l_out(n_timesteps,kernel_sizes,stride_sizes, 
                                                        max_pool_kernel_size=max_pool_kernel_size, 
                                                        max_pool_stride_size=max_pool_stride_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = l(x)
        return x


class CNNToTransformerEncoder(pl.LightningModule):
    def __init__(self, input_features, num_attention_heads, num_hidden_layers, n_timesteps, kernel_sizes=[5,3,1], out_channels = [256,128,64], 
                stride_sizes=[2,2,2], dropout_rate=0.3, num_labels=2, learning_rate=1e-3, warmup_steps=100,
                max_positional_embeddings = 1440*5, factor=64, inital_batch_size=100, clf_dropout_rate=0.0,
                **model_specific_kwargs) -> None:
        self.config = get_config_from_locals(locals())

        super(CNNToTransformerEncoder, self).__init__()
        
        self.d_model = out_channels[-1]
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps

        self.input_embedding = CNNEncoder(input_features, n_timesteps=n_timesteps, kernel_sizes=kernel_sizes,
                                out_channels=out_channels, stride_sizes=stride_sizes)
        
        if self.input_embedding.final_output_length < 1:
            raise ValueError("CNN final output dim is <1 ")                                
            
        self.positional_encoding = modules.PositionalEncoding(self.d_model, max_positional_embeddings)
        self.blocks = nn.ModuleList([
            modules.EncoderBlock(self.d_model, num_attention_heads, dropout_rate) for _ in range(num_hidden_layers)
        ])
        
        self.dense_interpolation = modules.DenseInterpolation(self.input_embedding.final_output_length, factor)
        self.clf = modules.ClassificationModule(self.d_model, factor, num_labels,
                                                dropout_p=clf_dropout_rate)
        
        self.criterion = build_loss_fn(model_specific_kwargs)

        self.name = "CNNToTransformerEncoder"
        self.base_model_prefix = self.name
        
        self.train_probs = []
        self.train_labels = []
        
        self.eval_probs = []
        self.eval_labels =[]

        self.batch_size = inital_batch_size
        self.train_dataset = None
        self.eval_dataset=None


    def forward(self, inputs_embeds,labels):
        x = inputs_embeds.transpose(1, 2)
        x = self.input_embedding(x)
        x = x.transpose(1, 2)
        x = self.positional_encoding(x)

        for l in self.blocks:
            x = l(x)
        
        x = self.dense_interpolation(x)
        logits = self.clf(x)
        loss =  self.criterion(logits,labels)
        return loss, logits
    
    def set_train_dataset(self,dataset):
        self.train_dataset = dataset
    
    def set_eval_dataset(self,dataset):
        self.eval_dataset = dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True,
                         shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.eval_dataset, batch_size=3*self.batch_size, pin_memory=True,
                          shuffle=True)

    def training_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        x = batch["inputs_embeds"]
        y = batch["label"]
        loss,logits = self.forward(x,y)
        
    
        self.log("train/loss", loss.item(),on_step=True, sync_dist=True)

        probs = torch.nn.functional.softmax(logits,dim=1)[:,-1]
        # self.train_probs.append(probs.detach().cpu())
        # self.train_labels.append(y.detach().cpu())
        return {"loss":loss, "preds": logits, "labels":y}

    def on_train_epoch_start(self):
        self.train_probs = []
        self.train_labels = []
        super().on_train_epoch_start()
    
    # def on_train_epoch_end(self):
    #     self.train_probs = []
    #     self.train_labels = []

    # def on_train_end(self):
    #     train_preds = torch.cat(self.train_probs, dim=0)
    #     train_labels = torch.cat(self.train_labels, dim=0)
    #     train_auc = torchmetrics.functional.auroc(train_preds,train_labels,pos_label=1)
    
    #     eval_preds = torch.cat(self.eval_probs, dim=0)
    #     eval_labels = torch.cat(self.eval_labels, dim=0)
        
    #     fpr, tpr, _ = torchmetrics.functional.roc(eval_preds, eval_labels, pos_label=1)
    #     eval_auc = torchmetrics.functional.auroc(eval_preds,eval_labels,pos_label=1)

    #     results = {"train/roc_auc":train_auc,
    #                 "eval/roc_auc":eval_auc}

    #     if check_for_wandb_run():
    #         labels = ["Positive"] * len(fpr)
    #         table = Table(columns= ["class","fpr","tpr"], data=list(zip(labels,fpr,tpr)))
    #         plot = wandb.plot_table(
    #             "wandb/area-under-curve/v0",
    #             table,
    #             {"x": "fpr", "y": "tpr", "class": "class"},
    #             {
    #                 "title": "ROC",
    #                 "x-axis-title": "False positive rate",
    #                 "y-axis-title": "True positive rate",
    #             },

    #         )
    #         results["eval/roc"] = plot
    #         wandb.log(results)

    #     super().on_train_end()
   
    def on_validation_epoch_start(self):
        self.eval_probs = []
        self.eval_labels = []
    
    def on_validation_epoch_end(self):
        eval_preds = torch.cat(self.eval_probs, dim=0)
        eval_labels = torch.cat(self.eval_labels, dim=0)
        eval_auc = torchmetrics.functional.auroc(eval_preds,eval_labels,pos_label=1)
        self.eval_probs = []
        self.eval_labels = []
    
        self.log_dict({"eval/roc_auc":eval_auc})
        # for obj in gc.get_objects():
        #     try:
        #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #             print(type(obj), obj.size())
        #     except:
        #         pass
        super().on_validation_epoch_end()

    def validation_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        x = batch["inputs_embeds"]
        y = batch["label"]
        with torch.no_grad():
            loss,logits = self.forward(x,y)

            self.log("eval/loss", loss.item(),on_step=True,sync_dist=True)
            
            probs = torch.nn.functional.softmax(logits,dim=1)[:,-1]
            self.eval_probs.append(probs.detach().cpu())
            self.eval_labels.append(y.detach().cpu())
        

        return {"loss":loss, "preds": logits, "labels":y}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
        on_tpu=False, using_native_amp=False, using_lbfgs=False,
    ):
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.learning_rate

        # update params
        optimizer.step(closure=optimizer_closure)
