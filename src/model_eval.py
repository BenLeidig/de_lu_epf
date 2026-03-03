from joblib import load

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

from .data_setup import EPFDataset, EPFDataModule
from .model_setup import TCN_LSTM_MHA


def willmotts_index(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    wi = 1 - (
        np.sum((y_true - y_pred) ** 2)
        / np.sum(
            (np.abs(y_pred - np.mean(y_true)) + (np.abs(y_true - np.mean(y_pred)))) ** 2
        )
    )
    return wi


def nash_sutcliffe_efficiency(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ns = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return ns


def legates_mccabes_index(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    lm = 1 - np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(y_true - np.mean(y_true)))
    return lm


def kling_gupta_efficiency(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cv_true = np.std(y_true) / np.mean(y_true)
    cv_pred = np.std(y_pred) / np.mean(y_pred)
    r = np.sum((y_true - y_true.mean()) * (y_pred - y_pred.mean())) / np.sqrt(
        np.sum((y_true - y_true.mean()) ** 2) * np.sum((y_pred - y_pred.mean()) ** 2)
    )
    kge = 1 - np.sqrt(
        (r - 1) ** 2
        + (np.mean(y_pred) / np.mean(y_true) - 1) ** 2
        + (cv_pred / cv_true) ** 2
    )
    return kge


def normalized_root_mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    nrmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred) / np.mean(y_true)
    return nrmse


def relative_mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmae = mean_absolute_error(y_true=y_true, y_pred=y_pred) / np.mean(y_true)
    return rmae


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    smape = (1 / len(y_true)) * np.sum(
        np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2)
    )
    return smape


def theils_inequality_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    numerator = np.sqrt((1 / n) * (np.sum(y_pred - y_true) ** 2))
    denominator = np.sqrt((1 / n) * np.sum(y_true**2)) + np.sqrt(
        (1 / n) * np.sum(y_pred**2)
    )
    tic = numerator / denominator
    return tic


def absolute_percentage_bias(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    apb = np.abs(np.sum(y_true - y_pred) / np.sum(y_true))
    return apb


def evaluate_model(y_true, y_pred) -> pd.DataFrame:
    """Provide a model evaluation table.

    Args:
        y_true (iterable): Actual response values.
        y_pred (iterable): Predicted response values.

    Returns:
        pd.DataFrame: Evaluation table.
    """

    abbr = [
        "r2",
        "wi",
        "ns",
        "lm",
        "kge",
        "rmse",
        "mae",
        "nrmse",
        "rmae",
        "smape",
        "tic",
        "apb",
    ]
    prop = ["variance"] * 5 + ["bias"] * 7
    metrics = [
        r2_score,
        willmotts_index,
        nash_sutcliffe_efficiency,
        legates_mccabes_index,
        kling_gupta_efficiency,
        root_mean_squared_error,
        mean_absolute_error,
        normalized_root_mean_squared_error,
        relative_mean_absolute_percentage_error,
        symmetric_mean_absolute_percentage_error,
        theils_inequality_coefficient,
        absolute_percentage_bias,
    ]
    results = [round(metric(y_true=y_true, y_pred=y_pred), 4) for metric in metrics]

    df = pd.DataFrame(
        data=list(zip(prop, abbr, results)), columns=["property", "metric", "score"]
    )

    return df


def get_params(best_params: dict):
    batch_size = best_params.pop("batch_size")
    hidden_sizes = [
        best_params.pop("hidden_size0"),
        best_params.pop("hidden_size1"),
        best_params.pop("hidden_size2"),
    ]
    lstm_dropouts = [best_params.pop("lstm_dropout0"), best_params.pop("lstm_dropout1")]
    channel_sizes = []
    channel_size_keys = list(best_params.keys())
    for s in channel_size_keys:
        if "channel_size_" in s:
            channel_sizes.append(best_params.pop(s))
    params = best_params.copy()
    params["hidden_sizes"] = hidden_sizes
    params["lstm_dropouts"] = lstm_dropouts
    params["channel_sizes"] = channel_sizes
    return batch_size, params


def get_fitted(
    imf: int,
    best_params: dict,
    data_dir: str,
    max_epochs: int = 100,
    accelerator="auto",
):

    batch_size, params = get_params(best_params=best_params)

    X_train_val = pd.read_pickle(data_dir + "df_train_val_scaled.pkl").to_numpy()
    y_train_val = pd.read_pickle(data_dir + "df_train_val_imf_scaled.pkl").to_numpy()
    X_train_val = np.concatenate([X_train_val, y_train_val], axis=1)
    test_dataset = EPFDataset(
        torch.tensor(X_train_val, dtype=torch.float32),
        torch.tensor(y_train_val[:, imf - 1], dtype=torch.float32),
    )
    train_val_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    )

    model = TCN_LSTM_MHA(input_size=16, **params)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(model, train_dataloaders=train_val_loader)

    return batch_size, model, trainer


def get_preds(
    imf: int, study_dir: str, data_dir: str, max_epochs: int = 100, accelerator="auto"
):

    imf_str = "resid" if imf == 6 else "imf" + str(imf)
    best_params = load(study_dir + imf_str + "_tcn_lstm_mha_study.pkl").best_params

    batch_size, model, trainer = get_fitted(
        imf=imf,
        best_params=best_params,
        data_dir=data_dir,
        max_epochs=max_epochs,
        accelerator=accelerator,
    )

    ## train_val preds
    X_train_val = pd.read_pickle(data_dir + "df_train_val_scaled.pkl").to_numpy()
    y_train_val = pd.read_pickle(data_dir + "df_train_val_imf_scaled.pkl").to_numpy()
    X_train_val = np.concatenate([X_train_val, y_train_val], axis=1)
    test_dataset = EPFDataset(
        torch.tensor(X_train_val, dtype=torch.float32),
        torch.tensor(y_train_val[:, imf - 1], dtype=torch.float32),
    )
    train_val_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    )
    tain_val_preds = trainer.predict(model, train_val_loader)
    flat_list = [
        item for sublist in tain_val_preds for inner in sublist for item in inner
    ]
    y_train_val_scaled_pred = np.array(flat_list)

    ## test preds
    X_test = pd.read_pickle(data_dir + "df_test_scaled.pkl").to_numpy()
    y_test = pd.read_pickle(data_dir + "df_test_imf_scaled.pkl").to_numpy()
    X_test = np.concatenate([X_test, y_test], axis=1)
    test_dataset = EPFDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test[:, imf - 1], dtype=torch.float32),
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
    )
    test_preds = trainer.predict(model, test_loader)
    flat_list = [item for sublist in test_preds for inner in sublist for item in inner]
    y_test_scaled_pred = np.array(flat_list)

    return y_train_val_scaled_pred, y_test_scaled_pred


def get_all_preds(
    study_dir: str,
    data_dir: str,
    scaler_dir: str,
    max_epochs: int = 100,
    accelerator="auto",
):
    scaler = load(scaler_dir + "ss_df_train_imf.pkl")

    train_val_preds_scaled, test_preds_scaled = [], []
    for imf in range(1, 7):
        y_train_val_scaled_pred, y_test_scaled_pred = get_preds(
            imf=imf,
            study_dir=study_dir,
            data_dir=data_dir,
            max_epochs=max_epochs,
            accelerator=accelerator,
        )
        train_val_preds_scaled.append(y_train_val_scaled_pred)
        test_preds_scaled.append(y_test_scaled_pred)

    train_val_preds_scaled = pd.DataFrame(
        data=np.array([_ for _ in train_val_preds_scaled]).T,
        columns=["imf" + str(i) for i in range(1, 6)] + ["resid"],
    )
    train_val_preds = pd.DataFrame(
        data=scaler.inverse_transform(train_val_preds_scaled),
        columns=train_val_preds_scaled.columns,
    )

    test_preds_scaled = pd.DataFrame(
        data=np.array([_ for _ in test_preds_scaled]).T,
        columns=["imf" + str(i) for i in range(1, 6)] + ["resid"],
    )
    test_preds = pd.DataFrame(
        data=scaler.inverse_transform(test_preds_scaled),
        columns=test_preds_scaled.columns,
    )

    return train_val_preds, train_val_preds_scaled, test_preds, test_preds_scaled
