import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import math
from einops import rearrange

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TSModel(pl.LightningModule):

    def __init__(self, n_steps, n_features: int, n_classes: int, n_hidden: int, bidirection: bool = True, 
                batch_first: bool = True, num_layers: int = 1, lr: float = 1e-3):
        super().__init__()
        self.lstm               =   nn.LSTM(input_size = n_features, hidden_size = n_hidden, 
                                            num_layers = num_layers, batch_first = batch_first, bidirectional= bidirection)
        self.dropout            =   nn.Dropout(p = 0.5)
        self.lr                 =   lr  
        self.hidden_size        =   n_hidden
        self.n_layers           =   num_layers
        self.bidirection        =   bidirection
        self.hidden             =   None

        D                       =   2 if bidirection else 1
        # sins_embed              =   SinusoidalPosEmb(n_features* D * n_steps)
        # self.time_mlp           =   nn.Sequential(
        #     sins_embed,
        #     nn.Linear(n_features * D * n_steps, n_features * D * n_steps),
        #     nn.GELU()
        # )

        self.mlp                =   nn.Sequential(
            nn.Linear(n_steps * D * n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, x, t):
        D               = 2 if self.bidirection else 1
        hidden_state    = torch.randn(D * self.n_layers, x.shape[0], self.hidden_size, device=x.device)
        cell_state      = torch.randn(D * self.n_layers, x.shape[0], self.hidden_size, device = x.device)
        self.hidden     = (hidden_state, cell_state)

        if t is not None:
            time_embed = self.time_mlp(t)
 
        out, (h,c) = self.lstm(x, self.hidden)
        out             = rearrange(out, 'b s f -> b (s f)')
        out             = self.mlp(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr = 1e-3)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.forward(x, None)
        loss = F.mse_loss(out, y) 
        self.log('train_loss', loss, on_epoch=True)
        return loss

    # def validation_step(self, *args, **kwargs):
    #     pass

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        loss.backward()
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.step()