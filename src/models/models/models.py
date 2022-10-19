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

from typing import Dict, Tuple,  Union, Any, Optional, List, Callable
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet as BaseResNet
from torchvision.models.resnet import BasicBlock

from sktime.classification.hybrid import HIVECOTEV2 as BaseHIVECOTEV2
import pandas as pd

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.cli import MODEL_REGISTRY

import torchmetrics

from src.models.losses import build_loss_fn
import src.models.models.modules as modules
from src.utils import get_logger, upload_pandas_df_to_wandb, binary_logits_to_pos_probs

from src.models.loops import DummyOptimizerLoop, NonNeuralLoop
from src.models.models.bases import ClassificationModel, NonNeuralMixin
from src.models.eval import (wandb_roc_curve, wandb_pr_curve, wandb_detection_error_tradeoff_curve,
                            classification_eval,  TorchMetricClassification, TorchMetricRegression, TorchMetricAutoencode)
from torch.utils.data.dataloader import DataLoader
from wandb.plot.roc_curve import roc_curve

logger = get_logger(__name__)


"""
 Helper functions:
"""

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


@MODEL_REGISTRY       
class CNNToTransformerClassifier(ClassificationModel):

    def __init__(self, num_attention_heads : int = 4, num_hidden_layers: int = 4,  
                kernel_sizes=[5,3,1], out_channels = [256,128,64], 
                stride_sizes=[2,2,2], dropout_rate=0.3, num_labels=2, 
                positional_encoding = False, pretrained_ckpt_path : Optional[str] = None,
                 **kwargs) -> None:

        super().__init__(**kwargs)

        if num_hidden_layers == 0:
            self.name = "CNNClassifier"
        else:
            self.name = "CNNToTransformerClassifier"
        n_timesteps, input_features = kwargs.get("input_shape")
        self.criterion = nn.CrossEntropyLoss()
        self.encoder = modules.CNNToTransformerEncoder(input_features, num_attention_heads, num_hidden_layers,
                                                      n_timesteps, kernel_sizes=kernel_sizes, out_channels=out_channels,
                                                      stride_sizes=stride_sizes, dropout_rate=dropout_rate, num_labels=num_labels,
                                                      positional_encoding=positional_encoding)
        
        self.head = modules.ClassificationModule(self.encoder.d_model, self.encoder.final_length, num_labels)

        if pretrained_ckpt_path:
            ckpt = torch.load(pretrained_ckpt_path)
            try:
                self.load_state_dict(ckpt['state_dict'])
            
            #TODO: Nasty hack for reverse compatability! 
            except RuntimeError:
                new_state_dict = {}
                for k,v in ckpt["state_dict"].items():
                    if not "encoder" in k :
                        new_state_dict["encoder."+k] = v
                    else:
                        new_state_dict[k] = v
                self.load_state_dict(new_state_dict, strict=False)

        self.save_hyperparameters()
        
    def forward(self, inputs_embeds,labels):
        encoding = self.encoder.encode(inputs_embeds)
        preds = self.head(encoding)
        loss =  self.criterion(preds,labels)
        return loss, preds

@MODEL_REGISTRY
class ResNet(ClassificationModel):
    
    def __init__(
        self,
        layers: List[int] = [2,2,2,2],
        num_classes: int = 2,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.name = "ResNet"
        self.base_model = BaseResNet(block=BasicBlock,
                                    layers=layers,
                                    num_classes=num_classes,
                                    groups=groups,
                                    width_per_group=width_per_group,
                                    replace_stride_with_dilation=replace_stride_with_dilation)
        
        # So that the model can handle the input shape...
        self.base_model.conv1 = nn.Conv2d(1, 64, 
                                          kernel_size=7, stride=2, padding=3, bias=False)                                    

        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()
    
    def forward(self, inputs_embeds,labels):
        x = inputs_embeds.unsqueeze(1) # Add a dummy dimension for channels
        preds = self.base_model(x)
        loss =  self.criterion(preds,labels)
        return loss, preds


@MODEL_REGISTRY
class TransformerClassifier(ClassificationModel):
    
    def __init__(
        self,
        num_attention_heads: int = 4,
        num_hidden_layers: int = 4,
        dropout_rate: float = 0.,
        num_classes: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.name = "TransformerClassifier"
        n_timesteps, input_features = kwargs.get("input_shape")

        self.criterion = nn.CrossEntropyLoss()
        self.blocks = nn.ModuleList([
            modules.EncoderBlock(input_features, num_attention_heads, dropout_rate) for _ in range(num_hidden_layers)
        ])
        
        self.head = modules.ClassificationModule(input_features, n_timesteps, num_classes)

    
    def forward(self, inputs_embeds,labels):
        x = inputs_embeds
        for l in self.blocks:
            x = l(x)

        preds = self.head(x)
        loss =  self.criterion(preds,labels)
        return loss, preds



        
@MODEL_REGISTRY
class HIVECOTE2(NonNeuralMixin,ClassificationModel):
    
    def __init__(
        self,
        n_jobs: int = -1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.base_model = BaseHIVECOTEV2(n_jobs=n_jobs)
        self.fit_loop = NonNeuralLoop()
        self.optimizer_loop = DummyOptimizerLoop()
        self.save_hyperparameters()
    
    def forward(self, inputs_embeds,labels):
        return self.base_model(inputs_embeds)