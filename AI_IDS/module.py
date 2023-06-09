import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding=0):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.layer(x)


class Stem(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            ConvBlock(1, 32, 3, 1, 1),
            ConvBlock(32, 32, 3, 1, 0),
            nn.MaxPool2d(3, 2, 0),
            ConvBlock(32, 64, 1, 1, 0),
            ConvBlock(64, 128, 3, 1, 1),
            ConvBlock(128, 128, 3, 1, 1)
        )

    def forward(self, x):
        return self.layer(x)
    

class InceptionA(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = ConvBlock(128, 32, 1, 1, 0)
        self.mid = nn.Sequential(
            ConvBlock(128, 32, 1, 1, 0),
            ConvBlock(32, 32, 3, 1, 1)
        )
        self.right = nn.Sequential(
            ConvBlock(128, 32, 1, 1, 0),
            ConvBlock(32, 32, 3, 1, 1),
            ConvBlock(32, 32, 3, 1, 1)
        )
        self.last = ConvBlock(96, 128, 1, 1, 0)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x_left = self.left(x)
        x_mid = self.mid(x)
        x_right = self.right(x)
        x_concat = torch.concat((x_left, x_mid, x_right), dim=1)
        x_res = self.last(x_concat)
        return self.activation(x + x_res)
    

class InceptionB(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = ConvBlock(448, 64, 1, 1, 0)
        self.right = nn.Sequential(
            ConvBlock(448, 64, 1, 1, 0),
            ConvBlock(64, 64, (1, 3), 1, (0, 1)),
            ConvBlock(64, 64, (3, 1), 1, (1, 0))
        )
        self.last = ConvBlock(128, 448, 1, 1, 0)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x_left = self.left(x)
        x_right = self.right(x)
        x_concat = torch.concat((x_left, x_right), dim=1)
        x_res = self.last(x_concat)
        return self.activation(x + x_res)


class ReductionA(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.MaxPool2d(3, 2, 0)
        self.mid = ConvBlock(128, 192, 3, 2, 0)
        self.right = nn.Sequential(
            ConvBlock(128, 96, 1, 1, 0),
            ConvBlock(96, 96, 3, 1, 1),
            ConvBlock(96, 128, 3, 2, 0)
        )
    
    def forward(self, x):
        x_left = self.left(x)
        x_mid = self.mid(x)
        x_right = self.right(x)
        return torch.concat((x_left, x_mid, x_right), dim=1)
    

class ReductionB(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.MaxPool2d(3, 2, 0)
        self.lmid = nn.Sequential(
            ConvBlock(448, 128, 1, 1, 0),
            ConvBlock(128, 192, 3, 2, 0),
        )
        self.rmid = nn.Sequential(
            ConvBlock(448, 128, 1, 1, 0),
            ConvBlock(128, 128, 3, 2, 0),
        )
        self.right = nn.Sequential(
            ConvBlock(448, 128, 1, 1, 0),
            ConvBlock(128, 128, 3, 1, 1),
            ConvBlock(128, 128, 3, 2, 0)
        )
    
    def forward(self, x):
        x_left = self.left(x)
        x_lmid = self.lmid(x)
        x_rmid = self.rmid(x)
        x_right = self.right(x)
        return torch.concat((x_left, x_lmid, x_rmid, x_right), dim=1)
