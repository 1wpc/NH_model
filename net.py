from typing import Tuple

import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, in_channels: int = 4, grid_size: Tuple[int, int] = (9, 9), num_classes: int = 2, dropout: float = 0.07):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.dropout = dropout

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.SELU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128), nn.SELU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(128), nn.SELU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(64), nn.SELU())
        self.conv5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64),nn.SELU())

        self.lin1 = nn.Sequential(
            nn.Linear(self.grid_size[0] * self.grid_size[1] * 64, 1024),
            nn.SELU(), # Not mentioned in paper
            nn.Dropout1d(self.dropout)
        )
        self.lin2 = nn.Sequential(
            nn.Linear(1024, 128),
            nn.SELU(),
            nn.Dropout1d(self.dropout)

        )
        self.lin3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.SELU(),
            nn.Dropout1d(self.dropout)
        )
        self.lin4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.SELU()
        )
        self.lin5 = nn.Sequential(
            nn.Linear(32, 16),
            nn.SELU()
        )
        self.lin6 = nn.Sequential(
            nn.Linear(16, 16),
            nn.SELU()

        )
        self.lin7 = nn.Sequential(
            nn.Linear(16, self.num_classes),
            # nn.Softmax(dim=1)

        )
        

    def feature_dim(self):
        return self.grid_size[0] * self.grid_size[1] * 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.flatten(start_dim=1)

        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        x = self.lin4(x)
        x = self.lin5(x)
        x = self.lin6(x)
        x = self.lin7(x)

        return x