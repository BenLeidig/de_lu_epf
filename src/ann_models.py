import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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


def create_nn_objective(model_class, cv, scaler_class, X_train_val:np.array, y_train_val:np.array, search_space:dict):
    '''
    Docstring for create_nn_objective
    
    :param model_class: ANN class.
    :param search_space: Should be in format {hidden_size:{method:suggest_int, ...}, ..., lr:0.01, batch_size:16} except when `sampler`='deterministic'.
    :type search_space: dict
    :param cv: Instantiated CV object.
    :param scaler_class: Scaler class.
    :param X_train_val: Feature matrix of train and validation set values.
    :type X_train_val: np.array
    :param y_train_val: Target array of train and validation set values.
    :type y_train_val: np.array
    '''
    def objective(trial):
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

        params = {}
        for name, spec in search_space.items():
            if isinstance(spec, dict):
                method = getattr(trial, spec['method'])
                kwargs = {k: v for k, v in spec.items() if k != 'method'}
                params[name] = method(name, **kwargs)
            else:
                params[name] = spec
        
        input_size = X_train_val.shape[1]
        fold_scores, step = [], 0
        for (train_idx, val_idx) in cv.split(X_train_val):
            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

            feature_scaler = scaler_class()
            X_train_scaled = feature_scaler.fit_transform(X_train)
            X_val_scaled = feature_scaler.transform(X_val)

            target_scaler = scaler_class()
            y_train_scaled = target_scaler.fit_transform(y_train)
            y_val_scaled = target_scaler.transform(y_val)

            X_train_scaled, y_train_scaled = create_sequences(X_train_scaled, y_train_scaled, seq_len=24*7, pred_len=24)
            X_val_scaled, y_val_scaled = create_sequences(X_val_scaled, y_val_scaled, seq_len=24*7, pred_len=24)

            X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
            y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
            X_val_scaled = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            y_val_scaled = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)

            lr, batch_size = params.pop('lr', 1e-3), params.pop('batch_size', 16)
            mod = model_class(input_size=input_size, **params).to(device)
            
            optimizer = torch.optim.Adam(mod.parameters(), lr=lr)
            criterion = nn.MSELoss()

            train_dataset = TensorDataset(X_train_scaled, y_train_scaled)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

            best_val_loss = np.inf
            best_state = None

            for epoch in range(100):
                mod.train()
                for xb_train_scaled, yb_train_scaled in train_loader:
                    optimizer.zero_grad()
                    yb_train_scaled_pred = mod(xb_train_scaled)
                    loss = criterion(yb_train_scaled_pred, yb_train_scaled)
                    loss.backward()
                    optimizer.step()
                
                mod.eval()
                with torch.no_grad():
                    y_val_scaled_pred = mod(X_val_scaled)
                    val_loss = criterion(y_val_scaled_pred, y_val_scaled).item()

                trial.report(val_loss, step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone().to('cpu') for k, v in mod.state_dict().items()}

                step += 1

            if best_state:
                mod.load_state_dict(best_state)
            y_val_pred = target_scaler.inverse_transform(y_val_scaled_pred.to(torch.device('cpu')).detach().numpy().flatten().reshape(-1, 1))
            val_loss = ((y_val_pred - y_val.detach().numpy())**2).mean()
            fold_scores.append(val_loss)

        return np.mean(fold_scores)
    
    return objective


class GRU(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=24, dropout=0.2):
        super().__init__()
        self.hidden_sizes = hidden_sizes

        self.gru1 = nn.GRU(input_size, hidden_sizes[0], batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.gru2 = nn.GRU(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.drop2 = nn.Dropout(dropout)

        self.gru3 = nn.GRU(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        self.drop3 = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_sizes[2], output_size)

    def forward(self, x):
        out, _ = self.gru1(x)
        out = self.drop1(out)

        out, _ = self.gru2(out)
        out = self.drop2(out)

        out, _ = self.gru3(out)
        out = self.drop3(out)

        out = out[:, -1, :]
        out = self.fc(out)
        return out.view(out.size(0), 24, 1)
    

def optimize_gru(X_train_val:np.array, y_train_val:np.array, path:str, batch_size:int=16):
    '''
    :param X_train_val: Feature matrix for the train and validation sets.
    :type X_train_val: np.array
    :param y_train_val: Target array for the train and validation sets.
    :type y_train_val: np.array
    :param path: File path location for the trained model state.
    :type path: str
    :param batch_size: Default is 16. Batch size for batch processing. 16 has best performance but 64 is the fastest.
    :type batch_size: int
    '''
    full_start = time.time()
    device = torch.device('mps')
    input_size = X_train_val.shape[1]
    counter = 0
    train_idx = np.arange(0, 14_232)
    val_idx = np.arange(14_232, 15_792)

    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)

    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train)
    y_val_scaled = target_scaler.transform(y_val)

    X_train_scaled, y_train_scaled = create_sequences(X_train_scaled, y_train_scaled, seq_len=24*7, pred_len=24)
    X_val_scaled, y_val_scaled = create_sequences(X_val_scaled, y_val_scaled, seq_len=24*7, pred_len=24)

    X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
    y_train_scaled = torch.tensor(y_train_scaled, dtype=torch.float32, device=device)
    X_val_scaled = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
    y_val_scaled = torch.tensor(y_val_scaled, dtype=torch.float32, device=device)

    mod = GRU(input_size=input_size).to(device)

    optimizer = torch.optim.Adam(mod.parameters())
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(X_train_scaled, y_train_scaled)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = np.inf
    best_state = None

    print('Training initiated.')
    for epoch in range(100):
        start = time.time()
        mod.train()
        for xb_train_scaled, yb_train_scaled in train_loader:
            optimizer.zero_grad()
            yb_train_scaled_pred = mod(xb_train_scaled)
            loss = criterion(yb_train_scaled_pred, yb_train_scaled)
            loss.backward()
            optimizer.step()
        end = time.time()

        mod.eval()
        with torch.no_grad():
            y_val_scaled_pred = mod(X_val_scaled)
            val_loss = criterion(y_val_scaled_pred, y_val_scaled).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone().to('cpu') for k, v in mod.state_dict().items()}
            counter = 0
        else:
            counter += 1

        print(f'\tEpoch {epoch} completed in {round(end-start, 2)} seconds.\n\t\tValidation MSE: {round(val_loss, 5)}\n\t\tCurrent best: {round(best_val_loss, 5)}')

        if counter >= 10:
            print(f'Patience limit reached at epoch {epoch}.')
            break

    if best_state:
        mod.load_state_dict(best_state)
    full_end = time.time()
    print(f'Training completed in {round(full_end-full_start, 2)} seconds.\n\tFinal model RMSE: {round(val_loss, 5)}')

    torch.save(mod.state_dict(), path)


# ## IMF 1, 2, 3
# class CNN_GRU(nn.Module):
#     def __init__(
#             self,
#             input_size,
#             hidden_size,
#             output_size,
#             conv1_out_channels, conv2_out_channels,
#             kernel_size1, kernel_size2,
#             num_layers=1,
#             dropout=0.0,
#         ):
#         super().__init__()

#         self.output_size = output_size
#         self.conv1 = nn.Conv1d(input_size, conv1_out_channels, kernel_size1, padding=1)
#         self.conv2 = nn.Conv1d(conv1_out_channels, conv2_out_channels, kernel_size2, padding=1)
#         self.bn1 = nn.BatchNorm1d(conv1_out_channels)
#         self.bn2 = nn.BatchNorm1d(conv2_out_channels)
#         self.relu = nn.ReLU()
#         self.gru = nn.GRU(conv2_out_channels, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
#         self.fc = nn.Linear(hidden_size, output_size*24)

#     def forward(self, x):
#         x = x.transpose(1, 2)
#         x = self.relu(self.bn1(self.conv1(x)))
#         x = self.relu(self.bn2(self.conv2(x)))
#         x = x.transpose(1, 2)
#         out, _ = self.gru(x)
#         out = self.fc(out[:, -1, :])
#         return out.view(out.size(0), 24, self.output_size)
    
# ## IMF 4, 5, 6, 7, 8
# class LSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
#         super().__init__()
#         self.output_size = output_size
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
#         self.fc = nn.Linear(hidden_size, output_size*24)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])
#         return out.view(out.size(0), 24, self.output_size)