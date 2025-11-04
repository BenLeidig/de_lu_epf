import os
import timeout_decorator
import joblib

import pandas as pd
import numpy as np

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone

from sklearn.svm import SVR
import optuna

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())


NETID = os.environ['USER']
SCRATCH_PATH = f'/scratch/{NETID}'
DATA_PATH = os.path.join(os.environ['HOME'], 'rfe_dataset_2019_2025.csv')
STORAGE_PATH = f'/scratch/{NETID}/svr_opt.db'

df = pd.read_csv(DATA_PATH, index_col='datetime')
datetime = pd.to_datetime(df.index, utc=True)
X = df.drop(columns='price')
y = df['price']
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

train_size = int(len(X_train_val) * (0.9*0.8))
step_size = 24*7*20
n_splits = (len(X_train_val) - train_size) // step_size
tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=train_size)

svr_pipe = Pipeline(
    steps=[
        ('normalizer', MinMaxScaler()),
        ('svr', SVR())
    ]
)
svr_pipe_ttr = TransformedTargetRegressor(
    regressor=svr_pipe,
    transformer=MinMaxScaler()
)

def objective(trial):

    C = trial.suggest_float('regressor__svr__C', 1e-6, 1e+6, log=True)
    gamma = trial.suggest_float('regressor__svr__gamma', 1e-6, 1e+6, log=True)

    mod = clone(svr_pipe_ttr)
    mod.set_params(regressor__svr__C=C, regressor__svr__gamma=gamma)

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_val)):

        X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
        y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

        mod.fit(X_train, y_train)
        
        y_val_pred = mod.predict(X_val)
        fold_scores.append(root_mean_squared_error(y_true=y_val, y_pred=y_val_pred))

        trial.report(np.mean(fold_scores), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
    return np.mean(fold_scores)

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=391),
    pruner=optuna.pruners.HyperbandPruner(),
    study_name='svr_opt_tpe_hyperband',
    storage=f'sqlite:///{STORAGE_PATH}',
    load_if_exists=True
)
study.optimize(objective, n_trials=256, n_jobs=16, timeout=60*60*24)
best_trial = study.best_trial
joblib.dump(study, os.path.join(SCRATCH_PATH, 'svr_opt_tpe_hyperband.pkl'))
joblib.dump(best_trial, os.path.join(SCRATCH_PATH, 'svr.pkl'))