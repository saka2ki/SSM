import torch
import torch.nn as nn
from .ssm import SSM

class StateSpaceModel(nn.Module):
  def __init__(self, vocab_size, dim, N, div, layer=1, dropout=0.):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, dim)
    self.layers = nn.ModuleList([
        nn.ModuleDict({
          'ssm':
            SSM(dim, N, div),
          'ffn':
            nn.Sequential(
                nn.Linear(dim, 4*dim),
                nn.GELU(),
                nn.Linear(4*dim, dim),
                nn.Dropout(dropout)
            ),
          'ln': nn.LayerNorm(dim),
    }) for _ in range(layer)])
    self.logits = nn.Linear(dim, vocab_size)
  def forward(self, x, cnn=True, is_emb=True, is_ssm=True, is_ffn=True):
    x = self.emb(x) if is_emb else x.unsqueeze(-1)
    for layer in self.layers:
      if is_ssm:
        x = layer['ssm'](x, cnn)# + x
      if is_ffn:
        x = layer['ffn'](x) + x
      x = layer['ln'](x)
    return self.logits(x)