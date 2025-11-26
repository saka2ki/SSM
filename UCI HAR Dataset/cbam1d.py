import torch 
import torch.nn as nn

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        x_max = torch.max(x,2)[0].unsqueeze(2)
        x_avg = torch.mean(x,2).unsqueeze(2)
        concat = torch.cat((x_max,x_avg), dim=2)
        output = self.conv(concat.transpose(1, 2)).transpose(1, 2)
        output = torch.sigmoid(output) * x 
        return output 

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        x_max = torch.max(x,1)[0]
        x_avg = torch.mean(x,1)
        linear_max = self.linear(x_max).unsqueeze(1)
        linear_avg = self.linear(x_avg).unsqueeze(1)
        output = linear_max + linear_avg
        output = torch.sigmoid(output) * x
        return output
    
class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x