import torch
from torch import nn

from module import Stem, InceptionA, InceptionB, ReductionA, ReductionB


class ReducedInceptionResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # in 29x29x1
        self.layer = nn.Sequential(
            Stem(), # 13x13x128 
            InceptionA(), # 13x13x128
            ReductionA(), # 6x6x448
            InceptionB(), # 6x6x448 
            ReductionB(), # 2x2x896
            nn.AvgPool2d(2, 1, 0), # 896
            nn.Flatten(),
            nn.Dropout1d(0.2),
            nn.Linear(896, 1), # 1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer(x).squeeze(-1)


class LSTMClassifier(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.lstm = nn.LSTM(83, self.hid_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hid_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        hidden = torch.zeros(x.shape[0], x.shape[1], self.hid_dim, requires_grad=True)
        cell = torch.zeros(x.shape[0], x.shape[1], self.hid_dim, requires_grad=True)
        out, _ = self.lstm(x, (hidden, cell))
        print(out[-1].shape)
        return self.classifier(out[:, -1])
        