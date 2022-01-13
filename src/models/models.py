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

from typing import Dict,  Union, Any, Optional
import os
from random import sample

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from petastorm.pytorch import BatchedDataLoader
from petastorm.reader import Reader

from pytorch_lightning.loggers.wandb import WandbLogger

from src.models.losses import build_loss_fn
from src.SAnD.core import modules
from src.utils import get_logger
from src.models.eval import (wandb_roc_curve, wandb_pr_curve, wandb_detection_error_tradeoff_curve,
                            classification_eval,  TorchMetricClassification, TorchMetricRegression)
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from wandb.plot.roc_curve import roc_curve

logger = get_logger(__name__)
class Config(object):
    def __init__(self,data) -> None:
        self.data = data
    def to_dict(self):
        return self.data


def conv_l_out(l_in,kernel_size,stride,padding=0, dilation=1):
    return int(np.floor((l_in + 2 * padding - dilation * (kernel_size-1)-1)/stride + 1))

def get_final_conv_l_out(l_in,kernel_sizes,stride_sizes,
                        max_pool_kernel_size=None, max_pool_stride_size=None):
    l_out = l_in
    for kernel_size, stride_size in zip(kernel_sizes,stride_sizes):
        l_out = conv_l_out(l_out,kernel_size,stride_size)
        if max_pool_kernel_size and max_pool_kernel_size:
            l_out = conv_l_out(l_out, max_pool_kernel_size,max_pool_stride_size)
    return int(l_out)

def convtrans_l_out(l_in,kernel_size,stride,padding=0, dilation=1):
    return (l_in -1) *  stride - 2 * padding + dilation * (kernel_size - 1)

def get_final_convtrans_l_out(l_in,kernel_sizes,stride_sizes,
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

        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.stride_sizes = stride_sizes
        self.max_pool_stride_size = max_pool_stride_size
        self.max_pool_kernel_size = max_pool_kernel_size

        super().__init__()
        self.input_features = input_features

        layers = []
        l_out = n_timesteps
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
            l_out = conv_l_out(l_out,kernel_sizes[i],stride_sizes[i])
            if max_pool_stride_size and max_pool_kernel_size:
                l_out = conv_l_out(l_out, max_pool_kernel_size,max_pool_stride_size)
                layers.append(nn.MaxPool1d(max_pool_kernel_size, stride=max_pool_stride_size))
            layers.append(nn.LayerNorm([out_channels[i],l_out]))
        self.layers = nn.ModuleList(layers)
        self.final_output_length = get_final_conv_l_out(n_timesteps,kernel_sizes,stride_sizes, 
                                                        max_pool_kernel_size=max_pool_kernel_size, 
                                                        max_pool_stride_size=max_pool_stride_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = l(x)
        return x

class CNNDecoder(nn.Module):
    def __init__(self, input_features, input_length,
                kernel_sizes=[1], out_channels = [128], 
                stride_sizes=[1], max_pool_kernel_size = 3,
                max_pool_stride_size=2) -> None:

        n_layers = len(kernel_sizes)
        assert len(out_channels) == n_layers
        assert len(stride_sizes) == n_layers

        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.stride_sizes = stride_sizes
        self.max_pool_stride_size = max_pool_stride_size
        self.max_pool_kernel_size = max_pool_kernel_size

        super(CNNDecoder,self).__init__()
        self.input_features = input_features

        layers = []
        for i in range(n_layers):
            if i == 0:
                in_channels = input_features
            else:
                in_channels = out_channels[i-1]

            # if i == n_layers - 1:
            #     out_channels = 2
            layers.append(nn.ConvTranspose1d(in_channels = in_channels,
                                    out_channels = out_channels[i],
                                    kernel_size=kernel_sizes[i],
                                    stride = stride_sizes[i]))
            layers.append(nn.ReLU())
            if max_pool_stride_size and max_pool_kernel_size:
                layers.append(nn.MaxPool1d(max_pool_kernel_size, stride=max_pool_stride_size))
        self.layers = nn.ModuleList(layers)
        self.final_output_length = get_final_conv_l_out(input_length,kernel_sizes,stride_sizes, 
                                                        max_pool_kernel_size=max_pool_kernel_size, 
                                                        max_pool_stride_size=max_pool_stride_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in self.layers:
            x = l(x)
        return x

    @staticmethod
    def from_inverse_of_encoder(encoder):
        if encoder.max_pool_stride_size or encoder.max_pool_kernel_size:
            raise ValueError("Can't invert encoders with MaxPool")
        else:
            print
            if isinstance(encoder,CNNEncoder):
                print("In instance ")
                # decoder_out_channels = encoder.out_channels[::-1]
                return CNNDecoder(
                    encoder.out_channels[-1],
                    encoder.final_output_length,
                    out_channels = encoder.out_channels[:-1][::-1] + [encoder.input_features],
                    stride_sizes = encoder.stride_sizes[::-1],
                    kernel_sizes = encoder.kernel_sizes[::-1],
                    max_pool_kernel_size=None,
                    max_pool_stride_size=None
                )
            
class CNNToTransformerEncoder(pl.LightningModule):
    def __init__(self, input_features, num_attention_heads, num_hidden_layers, n_timesteps, kernel_sizes=[5,3,1], out_channels = [256,128,64], 
                stride_sizes=[2,2,2], dropout_rate=0.3, num_labels=2, learning_rate=1e-3, warmup_steps=100,
                max_positional_embeddings = 1440*5, factor=64, inital_batch_size=100, clf_dropout_rate=0.0,
                train_mix_positives_back_in=False, train_mixin_batch_size=3, skip_cnn=False, wandb_id=None, 
                positional_encoding = False, model_head="classification",
                **model_specific_kwargs) -> None:
        
        self.config = get_config_from_locals(locals())
        super(CNNToTransformerEncoder, self).__init__()

        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.input_dim = (n_timesteps,input_features)

        self.input_embedding = CNNEncoder(input_features, n_timesteps=n_timesteps, kernel_sizes=kernel_sizes,
                                out_channels=out_channels, stride_sizes=stride_sizes)
        
        if not skip_cnn:
            self.d_model = out_channels[-1]
            final_length = self.input_embedding.final_output_length
        else:
            self.d_model = input_features
            final_length = n_timesteps

        if self.input_embedding.final_output_length < 1:
            raise ValueError("CNN final output dim is <1 ")                                
        
        if positional_encoding:
            self.positional_encoding = modules.PositionalEncoding(self.d_model, final_length)
        else:
            self.positional_encoding = None

        self.blocks = nn.ModuleList([
            modules.EncoderBlock(self.d_model, num_attention_heads, dropout_rate) for _ in range(num_hidden_layers)
        ])
        
        # self.dense_interpolation = modules.DenseInterpolation(final_length, factor)
        if model_head == "classification":
            self.head = modules.ClassificationModule(self.d_model, final_length, num_labels,
                                                    dropout_p=clf_dropout_rate)
            metric_class = TorchMetricClassification

        elif model_head == "regression":
            self.head = modules.RegressionModule(self.d_model, final_length, num_labels)
            metric_class = TorchMetricRegression

        self.train_metrics = metric_class(bootstrap_cis=False, prefix="train/")
        self.eval_metrics = metric_class(bootstrap_cis=True, prefix="eval/")
        self.test_metrics = metric_class(bootstrap_cis=True, prefix="test/")
        
        self.provided_train_dataloader = None
        
        self.criterion = build_loss_fn(model_specific_kwargs, task_type=model_head)
        if num_attention_heads > 0:
            self.name = "CNNToTransformerEncoder"
        else:
            self.name = "CNN"
            
        self.base_model_prefix = self.name
        
        self.train_probs = []
        self.train_labels = []
        self.train_mix_positives_back_in = train_mix_positives_back_in
        self.train_mixin_batch_size = train_mixin_batch_size
        self.positive_cache = []
        
        self.eval_probs = []
        self.eval_labels =[]

        self.test_probs = []
        self.test_labels =[]
        self.test_participant_ids = []
        self.test_dates = []
        self.test_losses = []

        self.batch_size = inital_batch_size
        self.train_dataset = None
        self.eval_dataset=None

        
        self.skip_cnn = skip_cnn
        self.save_hyperparameters()


    def forward(self, inputs_embeds,labels):
        encoding = self.encode(inputs_embeds)
        preds = self.head(encoding)
        loss =  self.criterion(preds,labels)
        return loss, preds

    def encode(self, inputs_embeds):
        if not self.skip_cnn:
            x = inputs_embeds.transpose(1, 2)
            x = self.input_embedding(x)
            x = x.transpose(1, 2)
        else:
            x = inputs_embeds
        
        if self.positional_encoding:
            x = self.positional_encoding(x)

        for l in self.blocks:
            x = l(x)
        
        # x = self.dense_interpolation(x)
        return x

    def set_train_dataset(self,dataset):
        self.train_dataset = dataset
    
    def set_eval_dataset(self,dataset):
        self.eval_dataset = dataset

    def train_dataloader(self):
        if isinstance(self.train_dataset,Reader):
            return BatchedDataLoader(self.train_dataset, batch_size=self.batch_size)
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size, pin_memory=True)
    
    def val_dataloader(self):
        if isinstance(self.eval_dataset, Reader):
            return BatchedDataLoader(self.eval_dataset, batch_size=3*self.batch_size)
        else:
            return DataLoader(self.eval_dataset, batch_size=3*self.batch_size, pin_memory=True)
    
    def on_train_start(self) -> None:

        self.train_metrics.apply(lambda x: x.to(self.device))
        self.eval_metrics.apply(lambda x: x.to(self.device))
        return super().on_train_start()
        
    def training_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        
        if not isinstance(batch,list):
            batch = [batch]

        x = torch.cat([x["inputs_embeds"] for x in batch],axis=0).type(torch.cuda.FloatTensor)
        y = torch.cat([x["label"] for x in batch], axis=0)

        if self.train_mix_positives_back_in and self.current_epoch > 0:
            self.positive_cache.extend(x[torch.where(y)].detach())
            if len(self.positive_cache) >= self.train_mixin_batch_size:
                pos_samples = [x[None,:,:] for x in sample(self.positive_cache,self.train_mixin_batch_size)]
                x = torch.cat([x] + pos_samples, axis=0)
                y = torch.cat([y, y.new([1] * self.train_mixin_batch_size)], axis=0)

        loss,logits = self.forward(x,y)
        
        self.log("train/loss", loss.item(),on_step=True)
        probs = torch.nn.functional.softmax(logits,dim=1)[:,-1]
        
        self.train_metrics.update(probs,y)

        self.train_probs.append(probs.detach())
        self.train_labels.append(y.detach())
        return {"loss":loss, "preds": logits, "labels":y}

    def on_train_epoch_end(self):
        train_preds = torch.cat(self.train_probs, dim=0)
        train_labels = torch.cat(self.train_labels, dim=0)
        # We get a DummyExperiment outside the main process (i.e. global_rank > 0)
        if os.environ.get("LOCAL_RANK","0") == "0":
            self.logger.experiment.log({"train/roc": wandb_roc_curve(train_preds,train_labels, limit = 9999)}, commit=False)
            self.logger.experiment.log({"train/pr": wandb_pr_curve(train_preds,train_labels)}, commit=False)
            self.logger.experiment.log({"train/det": wandb_detection_error_tradeoff_curve(train_preds,train_labels, limit=9999)}, commit=False)
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True)
        
        # Clean up for next epoch:
        self.train_metrics.reset()
        self.train_probs = []
        self.train_labels = []
        super().on_train_epoch_end()
    
    def on_train_epoch_start(self):
        self.train_metrics.to(self.device)
    
    def on_validation_epoch_start(self):
        torch.cuda.empty_cache()
        self.test_probs = []
    
    def on_test_epoch_end(self):


        test_preds = torch.cat(self.test_probs, dim=0)
        test_labels = torch.cat(self.test_labels, dim=0)

        # We get a DummyExperiment outside the main process (i.e. global_rank > 0)
        if os.environ.get("LOCAL_RANK","0") == "0":
            self.logger.experiment.log({"test/roc": wandb_roc_curve(test_preds,test_labels, limit = 9999)}, commit=False)
            self.logger.experiment.log({"test/pr": wandb_pr_curve(test_preds,test_labels)}, commit=False)
            self.logger.experiment.log({"test/det": wandb_detection_error_tradeoff_curve(test_preds,test_labels, limit=9999)}, commit=False)
        
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        
        # Clean up
        self.test_metrics.reset()
        self.test_probs = []
        self.test_labels = []
        
        super().on_validation_epoch_end()

    def predict_step(self, batch: Any) -> Any:
        x = batch["inputs_embeds"]
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
        
        loss,logits = self.forward(x,y)

        self.log("eval/loss", loss.item(),on_step=True,sync_dist=True)
        
        probs = torch.nn.functional.softmax(logits,dim=1)[:,-1]
        self.test_probs.append(probs.detach())
        self.test_labels.append(y.detach())
        
        self.test_metrics.update(probs,y)
        return {"loss":loss, "preds": logits, "labels":y}
        

    def on_validation_epoch_end(self):
        eval_preds = torch.cat(self.eval_probs, dim=0)
        eval_labels = torch.cat(self.eval_labels, dim=0)

        # We get a DummyExperiment outside the main process (i.e. global_rank > 0)
        if os.environ.get("LOCAL_RANK","0") == "0":
            self.logger.experiment.log({"eval/roc": wandb_roc_curve(eval_preds,eval_labels, limit = 9999)}, commit=False)
            self.logger.experiment.log({"eval/pr": wandb_pr_curve(eval_preds,eval_labels)}, commit=False)
            self.logger.experiment.log({"eval/det": wandb_detection_error_tradeoff_curve(eval_preds,eval_labels, limit=9999)}, commit=False)
        
        metrics = self.eval_metrics.compute()
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)
        
        # Clean up
        self.eval_metrics.reset()
        self.eval_probs = []
        self.eval_labels = []
        
        super().on_validation_epoch_end()

    def validation_step(self, batch, batch_idx) -> Union[int, Dict[str, Union[Tensor, Dict[str, Tensor]]]]:
        x = batch["inputs_embeds"].type(torch.cuda.FloatTensor)
        y = batch["label"]
        
        loss,logits = self.forward(x,y)

        self.log("eval/loss", loss.item(),on_step=True,sync_dist=True)
        
        probs = torch.nn.functional.softmax(logits,dim=1)[:,-1]
        self.eval_probs.append(probs.detach())
        self.eval_labels.append(y.detach())
        
        self.eval_metrics.update(probs,y)
        return {"loss":loss, "preds": logits, "labels":y}
    
    def configure_optimizers(self):
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

class CNNToTransformerAutoEncoder(pl.LightningModule):
    def __init__(self, input_features, num_attention_heads, num_hidden_layers, 
                    n_timesteps, kernel_sizes=[5, 3, 1], out_channels=[256, 128, 64], 
                    stride_sizes=[2, 2, 2], dropout_rate=0.3, num_labels=2, 
                    learning_rate=0.001, warmup_steps=100, max_positional_embeddings=1440 * 5,
                    factor=64, inital_batch_size=100, clf_dropout_rate=0, 
                    train_mix_positives_back_in=False, train_mixin_batch_size=3, 
                    **model_specific_kwargs) -> None:
        
        # super().__init__(input_features, num_attention_heads, num_hidden_layers, 
        #                 n_timesteps, kernel_sizes=kernel_sizes, out_channels=out_channels, 
        #                 stride_sizes=stride_sizes, dropout_rate=dropout_rate, 
        #                 num_labels=num_labels, learning_rate=learning_rate, 
        #                 warmup_steps=warmup_steps, max_positional_embeddings=max_positional_embeddings, 
        #                 factor=factor, inital_batch_size=inital_batch_size, 
        #                 clf_dropout_rate=clf_dropout_rate, train_mix_positives_back_in=train_mix_positives_back_in, 
        #                 train_mixin_batch_size=train_mixin_batch_size, **model_specific_kwargs)

        # Probably don't want to actually subclass
        self.encoder = CNNToTransformerEncoder(input_features, num_attention_heads, num_hidden_layers, 
                    n_timesteps, kernel_sizes=[5, 3, 1], out_channels=[256, 128, 64], 
                    stride_sizes=[2, 2, 2], dropout_rate=0.3, num_labels=2, 
                    learning_rate=0.001, warmup_steps=100, max_positional_embeddings=1440 * 5,
                    factor=64, inital_batch_size=100, clf_dropout_rate=0, 
                    train_mix_positives_back_in=False, train_mixin_batch_size=3, 
                    **model_specific_kwargs)

        self.decoder = CNN
        self.criterion = nn.MSELoss()
        self.name = "CNNToTransformerAutoEncoder"
        self.base_model_prefix = self.name

    def forward(self, inputs_embeds,labels):
        encoding = self.encoder.encode(inputs_embeds)
        

    
    def decode(inputs_embeds):
        ...