from cProfile import label
from turtle import forward
from typing import Tuple
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from src.models.models.bases import  ClassificationModel

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, input_shape: Tuple[int,int], 
                          patch_length: int,
                          dim: int,
                          num_hidden_layers: int,
                          num_attention_heads: int,
                          mlp_dim: int = 64,
                          pool: str = 'cls',
                          dim_head = 64,
                          dropout_rate: float = 0.,
                          emb_dropout_rate: float = 0.):

        super().__init__()
        n_timesteps, n_channels = input_shape
        assert n_timesteps % patch_length == 0, 'Input length must be divisible by the patch size.'


        num_patches = (n_timesteps // n_channels) 
        patch_dim = n_channels * patch_length
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            nn.BatchNorm1d(n_timesteps),
            Rearrange('b (l p) c -> b l (p c)', p = patch_length) ,
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout_rate)

        self.transformer = Transformer(dim, num_hidden_layers, num_attention_heads, dim_head, mlp_dim, dropout_rate)

        self.pool = pool
        self.to_latent = nn.Identity()


    
    def embed(self, x):

        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x
    
class ViTForClassification(ClassificationModel):
    
    def __init__(self, input_shape: Tuple[int,int], 
                       patch_length: int,
                       dim: int,
                       num_hidden_layers: int,
                       num_attention_heads: int,
                       mlp_dim: int = 64,
                       pool: str = 'cls',
                       dim_head = 64,
                       dropout_rate: float = 0.,
                       num_classes: int = 2,
                       emb_dropout_rate: float = 0.,
                       **kwargs):
        
        super().__init__(**kwargs)
        self.encoder = ViT(input_shape = input_shape,
                          patch_length = patch_length,
                          dim = dim,
                          num_hidden_layers = num_hidden_layers,
                          num_attention_heads = num_attention_heads,
                          mlp_dim = mlp_dim,
                          pool = pool,
                          dim_head = dim_head,
                          dropout_rate = dropout_rate,
                          emb_dropout_rate = emb_dropout_rate)

        # If we ever want to make the objective configurable we can do this:
        # https://pytorch-lightning.readthedocs.io/en/1.6.2/common/lightning_cli.html#class-type-defaults
        
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.objective = nn.CrossEntropyLoss()

    def forward(self, inputs_embeds, labels):
        encoding = self.encoder.embed(inputs_embeds)
        preds = self.head(encoding)
        loss = self.objective(preds,labels)
        return loss, preds

