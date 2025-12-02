import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, classes, dim, num_cnn=2, num_lstm=2):
        super().__init__()
        self.is_cnn, self.is_lstm = False, False
        self.input = nn.Sequential(
            nn.Conv1d(9, dim, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(dim)
        )
        if num_cnn > 0:
            self.cnn = nn.ModuleList([
                nn.ModuleDict({
                    'double_dim': nn.Sequential(
                        nn.Conv1d(dim*2**i, dim*2**(i+1), 3, padding=0),
                        nn.ReLU(),
                        nn.BatchNorm1d(dim*2**(i+1))),
                    'kernel_3': nn.Sequential(
                        nn.Conv1d(dim*2**(i+1), dim*2**(i+1), 3, padding=1),
                        nn.ReLU(),
                        nn.BatchNorm1d(dim*2**(i+1))),
                    'kernel_1': nn.Sequential(
                        nn.Conv1d(dim*2**(i+1), dim*2**(i+1), 1, padding=0),
                        nn.ReLU(),
                        nn.BatchNorm1d(dim*2**(i+1)))
                })
                for i in range(num_cnn)
            ])
            self.is_cnn=True
        if num_lstm > 0:
            self.lstm = nn.LSTM(dim*2**num_cnn, dim*2**(num_cnn), num_layers=num_lstm, batch_first=True, dropout=0.2, bidirectional=True)
            self.is_lstm=True
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(in_features=dim*2**num_cnn*2**self.is_lstm, out_features=classes)
        )

    def forward(self, x):
        x = self.input(x.transpose(1, 2))
        if self.is_cnn:
            for cnn in self.cnn:
                x = cnn['double_dim'](x)
                x = torch.max_pool1d(cnn['kernel_3'](x) + cnn['kernel_1'](x), kernel_size=2)
        if self.is_lstm:
            x, _ = self.lstm(x.transpose(1, 2))
            x = x.transpose(1, 2)
        return self.head(x)
                
        