import torch
import torch.nn as nn
#from .ssm import SSM
from s4 import SSM

class StateSpaceModel(nn.Module):
    def __init__(self, vocab_size, dim, N, layer=1, dropout=0.):
        super().__init__()
        #self.emb = nn.Linear(9, dim, bias=False)
        self.emb = nn.Conv1d(in_channels=9, out_channels=dim, kernel_size=3, stride=1, padding="same")
        self.layers = nn.ModuleList([
            nn.ModuleDict({
              'ssm':
                nn.Sequential(
                    SSM(dim, N),
                    nn.Dropout(dropout)
                ),
              'ffn':
                nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.GELU(),
                    nn.Linear(dim, dim),
                    nn.Dropout(dropout)
                ),
              'ln': nn.LayerNorm(dim),
        }) for _ in range(layer)])
        self.logits = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(dim, vocab_size, bias=True),
            #nn.Softmax(dim=-1)
        )
    
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        if isinstance(module, nn.LayerNorm) and module.bias is not None:
            nn.init.zeros_(module.bias)
        
    def forward(self, x, cnn=True):
        x = self.emb(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            x = layer['ssm'](x) + x
            x = layer['ffn'](x) + x
            x = layer['ln'](x)
        return self.logits(x.transpose(1, 2))