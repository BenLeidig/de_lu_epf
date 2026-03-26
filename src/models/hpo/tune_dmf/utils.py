import gc
from inspect import signature
from pathlib import Path

import optuna
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error


def create_dmf_data(set: str, features: list, target: str):
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
    data_path = BASE_DIR / "data/processed/dmf_data"
    df = pd.read_parquet(data_path / f"{set}_scaled.parquet")
    X = df[features]
    y = df[target]
    return X, y


def supports_parallel(model_class):
    return "n_jobs" in signature(model_class).parameters


def suggest_from_config(trial: optuna.trial.Trial, config: dict):
    """Obtain the suggested parameters for an Optuna trial given the provided configurations.
    An example configuration is:
    search_space = {
        "n_estimators": {"type": "int", "low": 100, "high": 1000},
        "max_depth": {"type": "int", "low": 3, "high": 12},
        "learning_rate": {"type": "float", "low": 1e-3, "high": 0.1, "log": True},
    }

    Args:
        trial (optuna.trial.Trial): Optuna trial to suggest parameters for.
        config (dict): Configurations for the parameter search spaces.

    Returns:
        dict: Dictionary of suggested parameter values. (i.e., {parameter1: value, parameter2:value, ...})
    """
    params = {}
    for name, specs in config.items():
        if specs["type"] == "int":
            params[name] = trial.suggest_int(name, int(specs["low"]), int(specs["high"]))
        elif specs["type"] == "float":
            params[name] = trial.suggest_float(
                name, float(specs["low"]), float(specs["high"]), log=specs.get("log", False)
            )
        elif specs["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, specs["choices"])
        elif specs["type"] == "constant":
            params[name] = specs["value"]
    return params


def create_dmf_objective(hour: int, model_class, search_space: dict, n_jobs: int):
    """Create the objective function for the provided model class forecasting the provided hour that searches the provided parameter space.
    An example configuration is:
    search_space = {
        "n_estimators": {"type": "int", "low": 100, "high": 1000},
        "max_depth": {"type": "int", "low": 3, "high": 12},
        "learning_rate": {"type": "float", "low": 1e-3, "high": 0.1, "log": True},
    }

    Args:
        hour (int): Hour of day to forecast (should be in 0 - 24, inclusively).
        model_class: Model class to instantiate, provided it has `.fit()` and `.predict()` methods.
        search_space (dict): Configurations for the parameter search spaces.

    Returns:
        Callable[[optuna.trial.Trial], float]: Objective function for an Optuna study.
    """
    BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
    cfg_path = BASE_DIR / "configs/models/preprocess_config.yaml"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["dmf"]

    features = cfg["features"]
    target = cfg["targets"][hour]

    X_train, y_train = create_dmf_data("train", features, target)
    X_val, y_val = create_dmf_data("val", features, target)

    def objective(trial: optuna.trial.Trial):
        params = suggest_from_config(trial, search_space)
        if supports_parallel(model_class):
            params["n_jobs"] = n_jobs

        mod = model_class(**params)
        mod.fit(X_train, y_train)
        y_val_pred = mod.predict(X_val)

        gc.collect()

        return mean_absolute_error(y_val, y_val_pred)

    return objective


def create_dmf_study(
    hour: int,
    model_class,
    search_space: dict,
    multivariate: bool = True,
    n_trials: int = 100,
    n_jobs: int = 1,
    random_state: int = 0,
):
    """Create an Optuna study for the provided hour, model class, and parameter space.
    An example configuration is:
    search_space = {
        "n_estimators": {"type": "int", "low": 100, "high": 1000},
        "max_depth": {"type": "int", "low": 3, "high": 12},
        "learning_rate": {"type": "float", "low": 1e-3, "high": 0.1, "log": True},
    }

    Args:
        hour (int):  Hour of day to forecast (should be in 0 - 24, inclusively).
        model_class: Model class to instantiate, provided it has `.fit()` and `.predict()` methods.
        search_space (dict): Configurations for the parameter search spaces.
        multivariate (bool, optional): If this is `True`, the multivariate TPE is used when suggesting parameters. The multivariate TPE is reported to outperform the independent TPE. Defaults to True.
        n_trials (int, optional): The number of trials for each process. `None` represents no limit in terms of the number of trials. The study continues to create trials until the number of trials reaches `n_trials`, `timeout` period elapses, `stop()` is called, or a termination signal such as SIGTERM or Ctrl+C is received.. Defaults to 100.
        random_state (int, optional): Seed for random number generator. Defaults to 0.

    Returns:
        optuna.study.Study: Optuna study after completed optimization.
    """
    objective = create_dmf_objective(hour, model_class, search_space, n_jobs)
    sampler = optuna.samplers.TPESampler(multivariate=multivariate, seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study
