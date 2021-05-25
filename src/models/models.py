from copy import copy

import torch
import torch.nn as nn

import numpy as np
from transformers import PretrainedConfig

from src.SAnD.core import modules
from src.models.losses import build_loss_fn

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


class CNNToTransformerEncoder(nn.Module):
    def __init__(self, input_features, num_attention_heads, num_hidden_layers, n_timesteps, kernel_sizes=[5,3,1], out_channels = [256,128,64], 
                stride_sizes=[2,2,2], dropout_rate=0.2, num_labels=2, 
                max_positional_embeddings = 1440*5, factor=64,
                **model_specific_kwargs) -> None:
        self.config = get_config_from_locals(locals())

        super(CNNToTransformerEncoder, self).__init__()
        
        self.d_model = out_channels[-1]

        self.input_embedding = CNNEncoder(input_features, n_timesteps=n_timesteps, kernel_sizes=kernel_sizes,
                                out_channels=out_channels, stride_sizes=stride_sizes)
        
        if self.input_embedding.final_output_length < 1:
            raise ValueError("CNN final output dim is <1 ")                                
            
        self.positional_encoding = modules.PositionalEncoding(self.d_model, max_positional_embeddings)
        self.blocks = nn.ModuleList([
            modules.EncoderBlock(self.d_model, num_attention_heads, dropout_rate) for _ in range(num_hidden_layers)
        ])
        
        self.dense_interpolation = modules.DenseInterpolation(self.input_embedding.final_output_length, factor)
        self.clf = modules.ClassificationModule(self.d_model, factor, num_labels)
        
        self.criterion = build_loss_fn(model_specific_kwargs)

        self.name = "CNNToTransformerEncoder"
        self.base_model_prefix = self.name

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

