import torch
import torch.nn as nn

class ResNet1d(nn.Module):
    def __init__(self, classes, dim):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(9, dim, kernel_size=7),
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim*2**i, dim*2**i, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm1d(dim*2**i),
                
                nn.Conv1d(dim*2**i, dim*2**(i+1), kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(dim*2**(i+1)),
                
                nn.Conv1d(dim*2**(i+1), dim*2**(i+1), kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm1d(dim*2**(i+1)),
            ) for i in range(4)        
        ])

        self.res = nn.ModuleList([
            nn.Conv1d(dim*2**i, dim*2**(i+1), kernel_size=1, stride=2)
            for i in range(4)        
        ])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(in_features=dim*2**4, out_features=classes)
        )

    def forward(self, x):
        x = self.input(x.transpose(1, 2))
        for conv, res in zip(self.conv, self.res):
            x = conv(x) + res(x)
        return self.head(x)