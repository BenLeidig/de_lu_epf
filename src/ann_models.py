import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def create_sequences(X, y, seq_len, pred_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len - pred_len + 1):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len:i+seq_len+pred_len])
    return np.array(Xs), np.array(ys)


## IMF 1, 2, 3
class CNN_GRU(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            conv1_out_channels, conv2_out_channels,
            kernel_size1, kernel_size2,
            num_layers=1,
            dropout=0.0,
        ):
        super().__init__()

        self.output_size = output_size
        self.conv1 = nn.Conv1d(input_size, conv1_out_channels, kernel_size1, padding=1)
        self.conv2 = nn.Conv1d(conv1_out_channels, conv2_out_channels, kernel_size2, padding=1)
        self.bn1 = nn.BatchNorm1d(conv1_out_channels)
        self.bn2 = nn.BatchNorm1d(conv2_out_channels)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(conv2_out_channels, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Linear(hidden_size, output_size*24)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out.view(out.size(0), 24, self.output_size)
    
## IMF 4, 5, 6, 7, 8
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Linear(hidden_size, output_size*24)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.view(out.size(0), 24, self.output_size)