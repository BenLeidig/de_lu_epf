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
datetime = pd.to_datetime(df.index, utc=True)
X = df.drop(columns='price')
y = df[['price']]

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

train_size = int(len(X_train_val) * (0.9*0.8))
step_size = 24*7*20
n_splits = (len(X_train_val) - train_size) // step_size
tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=train_size)


def create_sequences(X, y, seq_len, pred_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len - pred_len + 1):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len:i+seq_len+pred_len])
    return np.array(Xs), np.array(ys)


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


def objective(trial):
    slurm_procid = os.environ.get('SLURM_PROCID', '0')
    slurm_node = os.environ.get('SLURMD_NODENAME', 'unknown')
    print(f"Trial on node {slurm_node}, process {slurm_procid}")

    hidden_size = trial.suggest_int('hidden_size', 32, 256, log=True)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.7)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 4)

    fold_scores, patience, step = [], 10, 0
    for (train_idx, val_idx) in tscv.split(X_train_val):
        X_train, X_val = X_train_val.iloc[train_idx].to_numpy(), X_train_val.iloc[val_idx].to_numpy()
        y_train, y_val = y_train_val.iloc[train_idx].to_numpy(), y_train_val.iloc[val_idx].to_numpy()

        X_train, y_train = create_sequences(X_train, y_train, seq_len=24*7, pred_len=24)
        X_val, y_val = create_sequences(X_val, y_val, seq_len=24*7, pred_len=24)

        X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
        X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val = torch.tensor(y_val, dtype=torch.float32, device=device)

        mod = GRUnn(
            input_size=X_train.shape[2],
            hidden_size=hidden_size,
            output_size=y_train.shape[2],
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        mod.init_norm(X_train, y_train)
        X_train_norm = (X_train - mod.x_min) / (mod.x_max - mod.x_min)
        y_train_norm = (y_train - mod.y_min) / (mod.y_max - mod.y_min)
        X_val_norm = (X_val - mod.x_min) / (mod.x_max - mod.x_min)
        y_val_norm = (y_val - mod.y_min) / (mod.y_max - mod.y_min)
        
        optimizer = torch.optim.Adam(mod.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_dataset = TensorDataset(X_train_norm, y_train_norm)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataset = TensorDataset(X_val_norm, y_val_norm)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset)//10, shuffle=False)

        best_val_loss = np.inf
        patience_counter = 0
        best_state = None

        for epoch in range(100):
            mod.train()
            for xb_train_norm, yb_train_norm in train_loader:
                optimizer.zero_grad()
                yb_train_norm_pred = mod(xb_train_norm)
                loss = criterion(yb_train_norm_pred, yb_train_norm)
                loss.backward()
                optimizer.step()
            
            mod.eval()
            with torch.no_grad():
                val_loss = []
                for xb_val_norm, yb_val_norm in val_loader:
                    yb_val_norm_pred = mod(xb_val_norm)
                    batch_loss = criterion(yb_val_norm_pred, yb_val_norm).item()
                    val_loss.append(batch_loss)
                val_loss = np.mean(val_loss)

            trial.report(val_loss, step=step)
            step += 1
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = mod.state_dict().copy()

        if best_state:
            mod.load_state_dict(best_state)
        mod.eval()
        with torch.no_grad():
            curr_score = []
            for xb_val_norm, yb_val_norm in val_loader:
                yb_val_norm_pred = mod(xb_val_norm)
                yb_val_pred = mod.target_denorm(yb_val_norm_pred)
                yb_val = mod.target_denorm(yb_val_norm)
                curr_score.append(((yb_val_pred - yb_val) ** 2).mean().item())
            fold_scores.append(np.mean(curr_score))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return np.mean(fold_scores)

    
study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.HyperbandPruner(),
    study_name='gru_opt_tpe_hyperband_gpu'
)
study.optimize(objective, n_trials=250, timeout=60*60*48-60)
best_trial = study.best_trial
joblib.dump(study, os.path.join(SCRATCH_PATH, 'gru_tpe_hyperband_gpu.pkl'))
joblib.dump(best_trial, os.path.join(SCRATCH_PATH, 'gru.pkl'))