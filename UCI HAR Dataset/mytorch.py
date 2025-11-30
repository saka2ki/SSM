import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, classes, dim):
        super().__init__()
        self.lstm = nn.LSTM(9, dim, batch_first=True, bidirectional=True)
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
        self.conv1d = nn.Sequential(
            nn.Conv1d(9, dim, kernel_size=3),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=3),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(dim, dim*2, kernel_size=3),
            nn.BatchNorm1d(dim*2),
            nn.ReLU(),
            nn.Conv1d(dim*2, dim*2, kernel_size=3),
            nn.BatchNorm1d(dim*2),
            nn.ReLU(), 
            nn.MaxPool1d(kernel_size=2)
        )
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=dim*2, out_features=classes)
        )

    def forward(self, x):
        x = self.conv1d(x.transpose(1, 2))
        return self.head(x)    