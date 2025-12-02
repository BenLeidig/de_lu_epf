import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class CNN1_LSTMnn(nn.Module):

    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            conv1_out_channels,
            kernel_size,
            num_layers=1,
            dropout=0.0,
        ):
        super().__init__()

        self.output_size = output_size

        self.register_buffer('x_min', torch.zeros(1, 1, input_size))
        self.register_buffer('x_max', torch.ones(1, 1, input_size))

        self.register_buffer('y_min', torch.zeros(1, 1, output_size))
        self.register_buffer('y_max', torch.ones(1, 1, output_size))

        self.conv1 = nn.Conv1d(input_size, conv1_out_channels, kernel_size, padding=1)
        self.bn = nn.BatchNorm1d(conv1_out_channels)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(conv1_out_channels, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Linear(hidden_size, output_size*24)

    def init_norm(self, X_train, y_train):
        self.x_min = X_train.amin(dim=(0, 1), keepdim=True)
        self.x_max = X_train.amax(dim=(0, 1), keepdim=True)

        self.y_min = y_train.amin(dim=0, keepdim=True)
        self.y_max = y_train.amax(dim=0, keepdim=True)

    def target_denorm(self, y_pred):
        return y_pred * (self.y_max - self.y_min) + self.y_min

    def forward(self, x):
        x = (x - self.x_min) / (self.x_max - self.x_min)
        x = x.transpose(1, 2)
        x = self.relu(self.bn(self.conv1(x)))
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.view(out.size(0), 24, self.output_size)
    

class GRUnn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.output_size = output_size
        self.register_buffer('x_min', torch.zeros(1, 1, input_size))
        self.register_buffer('x_max', torch.ones(1, 1, input_size))
        self.register_buffer('y_min', torch.zeros(1, 1, output_size))
        self.register_buffer('y_max', torch.ones(1, 1, output_size))
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc = nn.Linear(hidden_size, output_size*24)

    def init_norm(self, X_train, y_train):
        self.x_min.copy_(X_train.amin(dim=(0, 1), keepdim=True))
        self.x_max.copy_(X_train.amax(dim=(0, 1), keepdim=True))
        self.y_min.copy_(y_train.amin(dim=(0, 1), keepdim=True))
        self.y_max.copy_(y_train.amax(dim=(0, 1), keepdim=True))

    def target_denorm(self, y_pred):
        return y_pred * (self.y_max - self.y_min) + self.y_min

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out.view(out.size(0), 24, self.output_size)