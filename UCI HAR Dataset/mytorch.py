import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, classes, dim):
        super().__init__()
        self.lstm = nn.LSTM(9, dim, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=dim*2, out_features=classes)
        )

    def forward(self, x):
        x, (h, c) = self.lstm(x)
        return self.head(x.transpose(1, 2))

class Conv1dClassifier(nn.Module):
    def __init__(self, classes, dim):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(9, dim, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(dim, dim*2, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(dim*2),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(dim*2, dim*4, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(dim*4),
            nn.MaxPool1d(kernel_size=2)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(in_features=dim*4, out_features=classes)
        )

    def forward(self, x):
        x = self.input(x.transpose(1, 2))

        return self.head(x)    