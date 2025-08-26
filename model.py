import torch
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
import chess
import pandas as pd
import config as cf
import random

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.2):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p)
        )
        # 1×1 projection if channels differ
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        out = self.conv(x)
        skip = self.shortcut(x)
        return F.relu(out + skip)

class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.block1 = ResidualBlock(7, 32, dropout_p=0.2)
        self.block2 = ResidualBlock(32, 64, dropout_p=0.2)
        self.block3 = ResidualBlock(64, 128, dropout_p=0.2)

        # Info NN
        self.info_fc1 = nn.Linear(13, 128)
        self.info_fc2 = nn.Linear(128, 64)

        # FC
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout_fc1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, board_tensor, info_tensor):

        # board_tensor: (batch, 7, 8, 8)
        # info_tensor:  (batch, 13, 1)

        x = self.block1(board_tensor)    # (batch, 32, 8, 8)

        # Info NN
        info = info_tensor.view(info_tensor.size(0), -1)      # (batch,13)
        info = F.relu(self.info_fc1(info))                    # (batch,128)
        info = F.relu(self.info_fc2(info))                    # (batch,64)
        info_bias = info.view(-1, 1, 8, 8)                    # (batch,1,8,8)

        x = x + info_bias.expand(-1, x.size(1), -1, -1)       # (batch,32,8,8)

        # Continue CNN
        x = self.block2(x)                                      # (batch, 64, 8, 8)
        x = self.block3(x)                                      # (batch,128, 8, 8)

        # FC
        x = x.view(x.size(0), -1)                               # (batch, 128*8*8)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        x = torch.sigmoid(self.fc2(x))                         # (batch,1)

        return x