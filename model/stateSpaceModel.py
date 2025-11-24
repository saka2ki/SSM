import torch
import torch.nn as nn
from .s4 import SSM

class StateSpaceModel(nn.Module):
  def __init__(self, vocab_size, dim, N, layer=1, dropout=0.):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, dim)
    self.layers = nn.ModuleList([
        nn.ModuleDict({
          'ssm':
            SSM(dim, N),
          'ffn':
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
                nn.Dropout(dropout)
            ),
          'ln': nn.LayerNorm(dim),
    }) for _ in range(layer)])
    self.logits = nn.Linear(dim, vocab_size, bias=True)

    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0, std=0.02)
    if isinstance(module, nn.LayerNorm) and module.bias is not None:
        nn.init.zeros_(module.bias)
        
  def forward(self, x, cnn=True, L=0):
    x = self.emb(x)
    for layer in self.layers:
      x = layer['ssm'](x, cnn, L=L) + x
      x = layer['ffn'](x) + x
      x = layer['ln'](x)
    return self.logits(x)