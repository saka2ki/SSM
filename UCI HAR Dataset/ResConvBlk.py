import torch
import torch.nn as nn

class ResConvBlk(nn.Module):
    def __init__(self, classes, dim):
        super().__init__()
        kernel = [5, 3, 3, 1, 3, 3, 1, 3, 3, 1]
        channel = [9, dim, dim*2, dim*2, dim*2, dim*4, dim*4, dim*4, dim*8, dim*8, dim*8]
        self.convblk = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    channel[i], 
                    channel[i+1], 
                    kernel_size=kernel[i],
                    stride = int(i < 1)+1,
                    padding = int(i%3==2)
                ),
                nn.ReLU(),
                nn.BatchNorm1d(channel[i+1])
            ) for i in range(10)
        ])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Dropout(0.6),
            nn.Linear(in_features=dim*8, out_features=classes),
            #nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # 最初の2層は単純に順次適用
        x = self.convblk[0](x.transpose(1, 2))

        for i in range(1, 10, 3):
            x = self.convblk[i](x)
            x = torch.max_pool1d(
                self.convblk[i+1](x) + self.convblk[i+2](x),
                kernel_size=2
            )  
        return self.head(x)
