import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

NETID = os.environ['USER']
SCRATCH_PATH = f'/scratch/{NETID}'
DATA_PATH = os.path.join(os.environ['HOME'], 'rfe_dataset_2019_2025.csv')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(DATA_PATH, index_col='datetime')
X = df.drop(columns='price')
y = df[['price']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

def create_sequences(X, y, seq_len, pred_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len - pred_len + 1):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len:i+seq_len+pred_len])
    return np.array(Xs), np.array(ys)

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
        self.x_min.copy_(X_train.amin(dim=(0, 1), keepdim=True))
        self.x_max.copy_(X_train.amax(dim=(0, 1), keepdim=True))
        self.y_min.copy_(y_train.amin(dim=(0, 1), keepdim=True))
        self.y_max.copy_(y_train.amax(dim=(0, 1), keepdim=True))

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
    
X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_len=24*7, pred_len=24)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len=24*7, pred_len=24)

X_train_seq = torch.tensor(X_train_seq, dtype=torch.float32, device=device)
y_train_seq = torch.tensor(y_train_seq, dtype=torch.float32, device=device)
X_test_seq = torch.tensor(X_test_seq, dtype=torch.float32, device=device)
y_test_seq = torch.tensor(y_test_seq, dtype=torch.float32, device=device)

mod = CNN1_LSTMnn(
    input_size=X_train_seq.shape[2],
    hidden_size=77,
    output_size=y_train_seq.shape[2],
    conv1_out_channels=77,
    kernel_size=7,
    num_layers=1,
    dropout=0.1644971956709153
).to(device)

mod.init_norm(X_train_seq, y_train_seq)
X_train_norm = (X_train_seq - mod.x_min) / (mod.x_max - mod.x_min)
y_train_norm = (y_train_seq - mod.y_min) / (mod.y_max - mod.y_min)
X_test_norm = (X_test_seq - mod.x_min) / (mod.x_max - mod.x_min)
y_test_norm = (y_test_seq - mod.y_min) / (mod.y_max - mod.y_min)

optimizer = torch.optim.Adam(mod.parameters(), lr=0.00012532274085524727)
criterion = nn.MSELoss()

train_dataset = TensorDataset(X_train_norm, y_train_norm)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_dataset = TensorDataset(X_test_norm, y_test_norm)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset)//10, shuffle=False)

best_test_loss = np.inf
best_state = None
patience = 20

for epoch in range(200):
    mod.train()
    for xb_train_norm, yb_train_norm in train_loader:
        optimizer.zero_grad()
        yb_train_norm_pred = mod(xb_train_norm)
        loss = criterion(yb_train_norm_pred, yb_train_norm)
        loss.backward()
        optimizer.step()
    
    mod.eval()
    with torch.no_grad():
        test_loss = []
        for xb_test_norm, yb_test_norm in test_loader:
            yb_test_norm_pred = mod(xb_test_norm)
            batch_loss = criterion(yb_test_norm_pred, yb_test_norm).item()
            test_loss.append(batch_loss)
        test_loss = np.mean(test_loss)
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_state = {k: v.clone() for k, v in mod.state_dict().items()}
        patience = 20
    else:
        patience -= 1

    if patience == 0:
        break

mod.load_state_dict(best_state)
torch.save(mod.state_dict(), os.path.join(SCRATCH_PATH, 'cnn_lstm_state.pt'))

# #!/bin/bash
# #SBATCH --job-name=cnn_lstm_final
# #SBATCH --output=/scratch/%u/cnn_lstm_%j.out
# #SBATCH --error=/scratch/%u/cnn_lstm_%j.err
# #SBATCH --account=stat
# #SBATCH --partition=stat
# #SBATCH --time=48:00:00
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH --mem=32G
# #SBATCH --gres=gpu:1

# source /u/bleidig2/venvs/epfvenv/bin/activate
# srun python /u/bleidig2/cnn_lstm_final_mod_training.py