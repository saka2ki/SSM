import torch
import torch.nn as nn
from cbam1d import CBAM

class CN_GR(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=3, out_channels=dim, kernel_size=3)
        self.tanh = nn.Tanh()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.gru = nn.GRU(input_size=dim, hidden_size=dim, num_layers=1, bidirectional=True)
        self.norm = nn.LayerNorm(normalized_shape=dim*2)
    def forward(self, x):
        x = self.conv(x.permute(0, 2, 1))
        x = self.tanh(x)
        x = self.maxpool(x).permute(0, 2, 1)
        x, _ = self.gru(x)
        return self.norm(x)

class ResCBAR(nn.Module):
    def __init__(self, classes, seq, dim):
        super().__init__()
        self.cn_gr1 = CN_GR(dim)
        self.cn_gr2 = CN_GR(dim)       
        self.cn_gr3 = CN_GR(dim)
        
        self.norm1 = nn.LayerNorm(normalized_shape=dim*2)
        self.cbam = CBAM(channels=dim*2, r=4)
        self.gru = nn.GRU(input_size=dim*2, hidden_size=dim*2, num_layers=1, bidirectional=True)
        self.norm2 = nn.LayerNorm(normalized_shape=dim*4)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(in_features=dim*4, out_features=classes),
            #nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.norm1(self.cn_gr1(x[:, :, 0:3]) + self.cn_gr2(x[:, :, 3:6]) + self.cn_gr3(x[:, :, 6:9]))
        x = self.cbam(x)
        x, _ = self.gru(x)
        x = self.norm2(x)
        return self.head(x.permute(0, 2, 1))