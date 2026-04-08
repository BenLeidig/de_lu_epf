import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pytorch_tcn import TCN
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data  # type: ignore

torch.set_float32_matmul_precision("high")


class DirectMultiStepForecaster(RegressorMixin, BaseEstimator):
    """Custom Direct Multi-Step Forecaster (DMF) wrapper class. Creates 24 models of the provided `model_class` class, each to forecast a specific hour of the next day.
    Args:
    params (dict): Dictionary of parameters with integer keys of 0 through 23 and dictionary values of `model_class` class parameters.
    model_class (_type_): Model class to use for all hours' models. Must have `.fit()` and `.predict()` methods.
    """

    def __init__(self, params: dict, model_class):
        self.params = params
        self.model_class = model_class

    def fit(self, X, Y):
        """Fit each hour's expert model. `X` and `Y` should be indexed daily, with each column in `Y` representing an hour of the next day to be forecasted.

        Args:
            X (array-like): Feature matrix to fit all expert models on. Should be indexed daily.
            Y (array-like): Target matrix to fit expert models on. Columns should be sorted ascending by hour of next day.

        Raises:
            ValueError: If the number of hyperparameter sets in `params` specified during initialization is not equal to the number of targets (should be 24).

        Returns:
            DirectMultiStepForecaster: self.
        """
        # n0 = X.shape[0] = Y.shape[0]
        # p = X.shape[1]
        # q = Y.shape[1] = len(self.params)
        # --------
        # X.shape = (n0, p)
        # Y.shape = (n0, q)
        # y_i.shape = (n0, )

        self.target_names_ = getattr(Y, "columns", None)
        X, Y = validate_data(self, X, Y, multi_output=True)
        self.models_ = {}

        if len(self.params) != Y.shape[1]:
            raise ValueError(
                "Number of hyperparameter sets must match number of targets"
            )

        for hour in range(Y.shape[1]):
            model = self.model_class(**self.params[hour])
            model.fit(X, Y[:, hour])
            self.models_[hour] = model

        return self

    def predict(self, X):
        # n = X.shape[0]
        # preds.shape = (n, q)

        check_is_fitted(self, "models_")
        idx = getattr(X, "index", None)
        X = validate_data(self, X, reset=False)

        preds = [self.models_[hour].predict(X) for hour in sorted(self.models_)]
        preds = np.column_stack(preds)

        if self.target_names_ is not None:
            return pd.DataFrame(preds, columns=self.target_names_, index=idx)

        return preds


class TCN_LSTM_MHA(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        channel_sizes: list,
        kernel_size: int,
        hidden_sizes: list,
        tcn_dropout: float,
        lstm_dropouts: list,
        mha_dropout: float,
        mha_heads: int,
        lr_init: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = torch.nn.HuberLoss()
        self.lr_init = lr_init

        #### model ####
        # ReLU
        self.relu = nn.ReLU()

        # TCN
        self.tcn = TCN(
            num_inputs=input_size,
            num_channels=channel_sizes,
            kernel_size=kernel_size,
            dropout=tcn_dropout,
            use_skip_connections=True,
            input_shape="NLC",  # (batch_size, time_steps, feature_channels)
        )

        # LSTM
        self.lstm0 = nn.LSTM(
            input_size=channel_sizes[-1], hidden_size=hidden_sizes[0], batch_first=True
        )
        self.dropout0 = nn.Dropout(lstm_dropouts[0])
        self.lstm1 = nn.LSTM(
            input_size=hidden_sizes[0], hidden_size=hidden_sizes[1], batch_first=True
        )
        self.dropout1 = nn.Dropout(lstm_dropouts[1])
        self.lstm2 = nn.LSTM(
            input_size=hidden_sizes[1], hidden_size=hidden_sizes[2], batch_first=True
        )

        # MHA
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden_sizes[-1],
            num_heads=mha_heads,
            dropout=mha_dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_sizes[-1])

        # head
        self.fc = nn.Linear(hidden_sizes[-1], 24)

    def forward(self, x):
        # TCN
        x = self.tcn(x)
        x = self.relu(x)

        # LSTM
        x, _ = self.lstm0(x)
        x = self.relu(x)
        x = self.dropout0(x)

        x, _ = self.lstm1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)

        # MHA
        mha_out, _ = self.mha(x, x, x)
        x = self.norm(x + mha_out)

        # head
        return self.fc(x[:, -1, :])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        self.log(
            "val_loss", val_loss, on_step=False, on_epoch=True, batch_size=x.size(0)
        )
        return val_loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_init)
        return optimizer
