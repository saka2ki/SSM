import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, dim, nhead, layer, dropout=0.):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 1024, dim) * 0.02)
        assert dim % nhead == 0
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim, 
            nhead=nhead,
            dim_feedforward=4*dim, 
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layer)
        self.head = nn.Linear(dim, vocab_size)
    def forward(self, x):
        B, L = x.shape
        x = self.token_emb(x) + self.pos_emb[:, :L]
        
        memory = torch.zeros_like(x)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(L)
        
        x = self.decoder(x, memory, tgt_mask=tgt_mask)
        return self.head(x)