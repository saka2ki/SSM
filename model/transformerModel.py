import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, dim, layer):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layer)
        self.head = nn.Linear(dim, vocab_size)
    def forward(self, x):
        x = x.transpose(0, 1)
        L, B = x.shape
        mask = torch.triu(torch.full((L, L), float('-inf')), diagonal=1)
        x = self.embedding(x)  # (seq_len, batch_size, dim)
        x = self.encoder(x, mask=mask)  # マスクを適用
        out = self.head(x)  # (seq_len, batch_size, vocab_size)
        return out.transpose(0, 1)