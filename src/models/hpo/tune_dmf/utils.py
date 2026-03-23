import gc
from pathlib import Path

import numpy as np
import optuna
import pandas as pd


def make_dmf_objective(trial: optuna.trial.Trial):
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
    data_path = BASE_DIR / "data/processed/dmf_data"

    df_train = pd.read_parquet(data_path / "train_scaled.parquet")
    df_test = pd.read_parquet(data_path / "test_scaled.parquet")
