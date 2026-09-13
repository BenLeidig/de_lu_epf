from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from joblib import load
from yaml import safe_load

from de_lu_epf.data.loading import ANNDataModule
from de_lu_epf.models.training import get_best_ann_params

## NOTE: a lot of this code is fairly readable and straightforward.
## Many code comments are not included hear for brevity and to improve
## readability. It should be noted, however, that date handling (in
## this source code) is set explicity for this research project.
## Reach out to ben.leidig@gmail.com if you have any questions.

# Maps a physical split name to the `dt_range` key in process_config.yaml
# that its data actually starts at ("train_val" starts wherever "train"
# does, since it's train+val concatenated).
_SPLIT_DATE_KEY = {
    "train": "train",
    "val": "val",
    "train_val": "train",
    "test": "test",
}


def _split_start(dt_range_cfg: dict, split: str) -> pd.Timestamp:
    """Resolve the real start date of a named split from process_config.yaml.

    Raises:
        ValueError: If `split` doesn't match a known split name.
    """
    key = split.strip().lower()
    if key not in _SPLIT_DATE_KEY:
        raise ValueError(
            f"Unknown split {split!r}. Valid options: {sorted(_SPLIT_DATE_KEY)}"
        )
    return pd.to_datetime(dt_range_cfg[_SPLIT_DATE_KEY[key]]["start"], utc=True)


def fetch_data(model_type: str, split: str):
    """Fetch the scaled and unscaled data for a given model type and split.

    Args:
        model_type (str): The type of model (e.g., "dmf", "hybrid").
        split (str): The data split to fetch (e.g., "train", "test").

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the scaled and unscaled dataframes.
    """

    BASE_DIR = Path(__file__).parent.parent.parent.parent
    DATA_DIR = BASE_DIR / "data/processed"
    df_scaled = pd.read_parquet(DATA_DIR / f"{model_type}/{split}_scaled.parquet")
    df = pd.read_parquet(DATA_DIR / f"{model_type}/{split}.parquet")
    return df_scaled, df


def fetch_features_targets(model_type: str):
    """Fetch the features and targets for a given model type from the preprocessing configuration.

    Args:
        model_type (str): The type of model (e.g., "dmf", "hybrid").

    Returns:
        Tuple[list, list]: A tuple containing the list of feature column names and the list of target column names.
    """

    BASE_DIR = Path(__file__).parent.parent.parent.parent
    CFG_DIR = BASE_DIR / "configs/models"
    with open(CFG_DIR / "preprocess_config.yaml") as f:
        cfg = safe_load(f)[model_type]
    return cfg["features"], cfg["targets"]


def fetch_fitted(model_type: str, model_name: str, final: bool = True):
    """Fetch a fitted model for a given model type and model name.

    Args:
        model_type (str): The type of model (e.g., "dmf", "hybrid").
        model_name (str): The name of the fitted model file (without the .pkl extension).
        final (bool): If True (default), fetch the single, already-selected
            winning model's train_val-refit (final) fit, stored under
            "full" (the only mode that existed for DMF before candidate/final
            were distinguished). If False, fetch the train-only-fitted
            "candidate" model, stored under "candidates", used to compare
            models via validation performance before a winner is selected.

    Returns:
        Any: The fitted model object loaded from the corresponding .pkl file.
    """
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    stage_dir = "full" if final else "candidates"
    MODEL_DIR = BASE_DIR / f"models/{model_type}/{stage_dir}"
    return load(MODEL_DIR / f"{model_name}.pkl")


def fetch_full_scalers(model_type: str):
    """Fetch the full feature and target scalers for a given model type.

    Args:
        model_type (str): The type of model (e.g., "dmf", "hybrid").

    Returns:
        Tuple[Any, Any]: A tuple containing the feature scaler and target scaler objects.
    """
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    MODEL_DIR = BASE_DIR / f"models/{model_type}/full"
    feature_scaler = load(MODEL_DIR / "feature_scaler.pkl")
    target_scaler = load(MODEL_DIR / "target_scaler.pkl")
    return feature_scaler, target_scaler


def get_predictions_dmf(
    model_name: str, train_split: str, test_split: str, final: bool = False
):
    """Get predictions for a DMF model.

    Args:
        model_name (str): The name of the fitted DMF model file (without the .pkl extension).
        train_split (str): The name of the training data split.
        test_split (str): The name of the testing data split.
        final (bool): If False (default), use the train-only-fitted
            "candidate" model (see `fetch_fitted`) — pass `train_split="train"`,
            `test_split="val"` to compare this model's validation performance
            against other candidates. If True, use the single, already-selected
            winning model's train_val-refit — pass `train_split="train_val"`,
            `test_split="test"` for its one-time final evaluation.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing predictions as DataFrames.
    """
    model_type = "dmf"

    df_train_scaled, _ = fetch_data(model_type=model_type, split=train_split)
    df_test_scaled, _ = fetch_data(model_type=model_type, split=test_split)

    features, _ = fetch_features_targets(model_type)
    X_train_scaled = df_train_scaled[features]
    X_test_scaled = df_test_scaled[features]

    dmf = fetch_fitted(model_type=model_type, model_name=model_name, final=final)

    Y_train_pred = dmf.predict(X_train_scaled)
    Y_test_pred = dmf.predict(X_test_scaled)

    train_val_idx = pd.date_range(
        start=X_train_scaled.index[0],
        periods=Y_train_pred.shape[0] * Y_train_pred.shape[1],
        freq="h",
        tz="UTC",
    )
    test_idx = pd.date_range(
        start=df_test_scaled.index[0],
        periods=Y_test_pred.shape[0] * Y_test_pred.shape[1],
        freq="h",
        tz="UTC",
    )

    y_train_pred = pd.DataFrame(
        data=np.asarray(Y_train_pred).flatten(),
        columns=["price"],
        index=train_val_idx,
    )
    y_test_pred = pd.DataFrame(
        data=np.asarray(Y_test_pred).flatten(),
        columns=["price"],
        index=test_idx,
    )

    return y_train_pred, y_test_pred


def get_predictions_hybrid(
    model_name: str,
    model_class,
    train_split: str,
    test_split: str,
    final: bool = False,
):
    """Get predictions for a hybrid model.

    Args:
        model_name (str): The name of the fitted hybrid model file (without the .ckpt extension).
        model_class (_type_): The class of the hybrid model.
        train_split (str): The name of the training data split.
        test_split (str): The name of the testing data split.
        final (bool): If False (default), use the "candidate" checkpoints
            under "full" (train/val-fitted, per `get_fitted_ann`) — pass
            `train_split="train"`, `test_split="val"` to compare this
            architecture's validation performance against other candidates.
            If True, use the single, already-selected winning model's
            train_val-refit checkpoints under "final" — pass
            `train_split="train_val"`, `test_split="test"` for its one-time
            final evaluation.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the training and testing predictions as NumPy arrays.
    """

    BASE_DIR = Path(__file__).parent.parent.parent.parent
    DATA_DIR = BASE_DIR / "data/processed/hybrid"
    CFG_DIR = BASE_DIR / "configs"
    MODEL_DIR = BASE_DIR / f"models/hybrid/{'final' if final else 'full'}"

    with open(CFG_DIR / "data/process_config.yaml") as f:
        dt_range_cfg = safe_load(f)["dt_range"]

    seq_len = 24 * 7 * 2
    train_pred_dict = {}
    test_pred_dict = {}
    targets = ["imf1", "imf2", "imf3", "imf4", "imf5", "imf_resid"]
    for target_col in targets:
        batch_size, params = get_best_ann_params(
            target_col=target_col, model_name=model_name, model_type="hybrid"
        )

        datamodule = ANNDataModule(
            data_dir=DATA_DIR, batch_size=batch_size, target_col=target_col
        )
        train_dataloader = datamodule.get_dataloader(train_split)
        test_dataloader = datamodule.get_dataloader(test_split)

        model = model_class.load_from_checkpoint(
            MODEL_DIR / f"{model_name}/{target_col}_{model_name}.ckpt",
            input_size=datamodule.input_size,
            **params,
        )
        model.eval()

        trainer = pl.Trainer(
            accelerator="auto",
            logger=False,
            enable_checkpointing=False,
        )

        y_train_pred = trainer.predict(model, dataloaders=train_dataloader)
        train_pred_dict[target_col] = (
            torch.cat(y_train_pred, dim=0).detach().cpu().numpy().reshape(-1)  # type: ignore
        )

        y_test_pred = trainer.predict(model, dataloaders=test_dataloader)
        test_pred_dict[target_col] = (
            torch.cat(y_test_pred, dim=0).detach().cpu().numpy().reshape(-1)  # type: ignore
        )

    # NOTE: derive each side's start date from the split actually requested,
    ## not a hardcoded "train"/"test" config key - otherwise a call with
    ## test_split="val" would silently get its rows stamped with fabricated
    ## "test" (2024) dates instead of its real (2023) ones.
    train_start = _split_start(dt_range_cfg, train_split) + pd.Timedelta(seq_len, "h")
    test_start = _split_start(dt_range_cfg, test_split) + pd.Timedelta(seq_len, "h")

    train_val_idx = pd.date_range(
        start=train_start,
        periods=len(train_pred_dict["imf1"]),
        freq="h",
    )

    test_idx = pd.date_range(
        start=test_start,
        periods=len(test_pred_dict["imf1"]),
        freq="h",
    )

    # NOTE: no year-based filtering here - each split's dataset only ever
    ## contains rows from that split's own date range, so the index computed
    ## above already reflects exactly the requested data. A hardcoded
    ## "year < 2024" / "year == 2024" filter only happened to be a no-op for
    ## the historical train_val/test case; it would silently empty out a
    ## "val" (2023) split's predictions.
    train_pred_df = pd.DataFrame(train_pred_dict)
    train_pred_df["datetime"] = train_val_idx
    train_pred_df = train_pred_df.set_index("datetime")

    test_pred_df = pd.DataFrame(test_pred_dict)
    test_pred_df["datetime"] = test_idx
    test_pred_df = test_pred_df.set_index("datetime")

    _, target_scaler = fetch_full_scalers(model_type="hybrid")
    train_pred_df = pd.DataFrame(
        data=target_scaler.inverse_transform(train_pred_df),
        columns=train_pred_df.columns,
        index=train_pred_df.index,
    )
    test_pred_df = pd.DataFrame(
        data=target_scaler.inverse_transform(test_pred_df),
        columns=test_pred_df.columns,
        index=test_pred_df.index,
    )

    train_pred_df["price"] = train_pred_df.sum(axis=1)
    test_pred_df["price"] = test_pred_df.sum(axis=1)

    return train_pred_df, test_pred_df


def get_predictions_ann(
    model_name: str,
    model_class,
    train_split: str,
    test_split: str,
    final: bool = False,
):
    """Get predictions for an ANN model.

    Args:
        model_name (str): The name of the fitted ANN model file (without the .ckpt extension).
        model_class (_type_): The class of the ANN model.
        train_split (str): The name of the training data split.
        test_split (str): The name of the testing data split.
        final (bool): If False (default), use the "candidate" checkpoint
            under "full" (train/val-fitted, per `get_fitted_ann`) — pass
            `train_split="train"`, `test_split="val"` to compare this
            architecture's validation performance against other candidates.
            If True, use the single, already-selected winning model's
            train_val-refit checkpoint under "final" — pass
            `train_split="train_val"`, `test_split="test"` for its one-time
            final evaluation.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing predictions as DataFrames.
    """

    BASE_DIR = Path(__file__).parent.parent.parent.parent
    DATA_DIR = BASE_DIR / "data/processed/ann"
    CFG_DIR = BASE_DIR / "configs"
    MODEL_DIR = BASE_DIR / f"models/ann/{'final' if final else 'full'}"

    with open(CFG_DIR / "data/process_config.yaml") as f:
        dt_range_cfg = safe_load(f)["dt_range"]

    seq_len = 24 * 7 * 2
    target_col = "price"
    train_pred_dict = {}
    test_pred_dict = {}
    batch_size, params = get_best_ann_params(
        target_col=target_col, model_name=model_name, model_type="ann"
    )

    datamodule = ANNDataModule(
        data_dir=DATA_DIR, batch_size=batch_size, target_col=target_col
    )
    train_dataloader = datamodule.get_dataloader(train_split)
    test_dataloader = datamodule.get_dataloader(test_split)

    model = model_class.load_from_checkpoint(
        MODEL_DIR / f"{model_name}.ckpt",
        input_size=datamodule.input_size,
        **params,
    )
    model.eval()

    trainer = pl.Trainer(
        accelerator="auto",
        logger=False,
        enable_checkpointing=False,
    )

    y_train_pred = trainer.predict(model, dataloaders=train_dataloader)
    train_pred_dict[target_col] = (
        torch.cat(y_train_pred, dim=0).detach().cpu().numpy().reshape(-1)  # type: ignore
    )

    y_test_pred = trainer.predict(model, dataloaders=test_dataloader)
    test_pred_dict[target_col] = (
        torch.cat(y_test_pred, dim=0).detach().cpu().numpy().reshape(-1)  # type: ignore
    )

    # NOTE: derive each side's start date from the split actually requested,
    ## not a hardcoded "train"/"test" config key - see get_predictions_hybrid
    ## for the same fix and rationale.
    train_start = _split_start(dt_range_cfg, train_split) + pd.Timedelta(seq_len, "h")
    test_start = _split_start(dt_range_cfg, test_split) + pd.Timedelta(seq_len, "h")

    train_idx = pd.date_range(
        start=train_start,
        periods=len(train_pred_dict[target_col]),
        freq="h",
    )

    test_idx = pd.date_range(
        start=test_start,
        periods=len(test_pred_dict[target_col]),
        freq="h",
    )

    # NOTE: no year-based filtering - see get_predictions_hybrid for why.
    train_pred_df = pd.DataFrame(train_pred_dict)
    train_pred_df["datetime"] = train_idx
    train_pred_df = train_pred_df.set_index("datetime")

    test_pred_df = pd.DataFrame(test_pred_dict)
    test_pred_df["datetime"] = test_idx
    test_pred_df = test_pred_df.set_index("datetime")

    _, target_scaler = fetch_full_scalers(model_type="ann")
    train_pred_df = pd.DataFrame(
        data=target_scaler.inverse_transform(train_pred_df),
        columns=train_pred_df.columns,
        index=train_pred_df.index,
    )
    test_pred_df = pd.DataFrame(
        data=target_scaler.inverse_transform(test_pred_df),
        columns=test_pred_df.columns,
        index=test_pred_df.index,
    )

    return train_pred_df, test_pred_df


def get_all_set_preds(set: str):
    """Get all predictions for a given dataset split.

    Args:
        set (str): The name of the dataset split (e.g., "train" or "test").

    Returns:
        pd.DataFrame: A DataFrame containing all model predictions for the specified dataset split.
    """

    BASE_DIR = Path(__file__).parent.parent.parent.parent
    DATA_DIR = BASE_DIR / "data"

    concat_list = []

    for dir in ["ann", "dmf", "hybrid"]:
        PREDS_DIR = DATA_DIR / f"predictions/{set}/{dir}"

        for f in PREDS_DIR.iterdir():
            df = pd.read_parquet(f)
            model_name = f.name.split(f"_{set}_")[0]
            preds = df[["price"]].rename(
                columns={"price": model_name.upper().replace("_", "-")}
            )  # type:ignore
            concat_list.append(preds)

    return pd.concat(concat_list, axis=1, join="inner")
