import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_imfs(df: pd.DataFrame):
    """Plot the original price alongside the IMFs and residual subsignals.

    Args:
        df (pd.DataFrame): DataFrame containing the columns `imf1`, `imf2`, `imf3`, `imf4`, `imf5`, `resid`, and `price`.

    Returns:
        fig (plt.Fig)
        axes (np.array)
    """

    fig, axes = plt.subplots(7, sharex=True, figsize=(12, 8), tight_layout=True)
    colors = plt.colormaps["tab10"].colors
    labels = ["Price", "IMF1", "IMF2", "IMF3", "IMF4", "IMF5", "Residual"]
    columns = ["price", "imf1", "imf2", "imf3", "imf4", "imf5", "resid"]

    for ax, c, lab, col in zip(axes, colors, labels, columns):
        ax.axhline(0, linewidth=0.7, color="black")
        ax.plot(df.index, df[col], linewidth=0.5, c=c, alpha=0.9)
        ax.fill_between(df.index, df[col], color=c, alpha=0.2)
        ax.grid(axis="x", linestyle="--")
        ax.set_ylabel(lab, fontsize=15)
        ax.tick_params(axis="x", labelsize=15)

    return fig, axes


def plot_residuals(df_actual: pd.DataFrame, df_pred: pd.DataFrame):
    """Plot the actual and predicted prices together and their residual.

    Args:
        df_actual (pd.DataFrame): DataFrame containing the actual column `price`.
        df_pred (pd.DataFrame): DataFrame containing the predicted column `price`.

    Returns:
        fig (plt.Fig)
        axes (np.array)
    """

    resid_price = df_actual[["price"]] - df_pred[["price"]]
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8), tight_layout=True)
    ax0, ax1 = axes[0], axes[1]

    ax0.axhline(0, linewidth=0.7, color="black")
    ax0.grid(axis="x", linestyle="--")
    ax0.set_ylabel("Price (€/MWHr)", fontsize=15)

    ax0.plot(
        df_actual.index,
        df_actual["price"],
        linewidth=0.3,
        c="tab:blue",
        alpha=0.9,
        label="Actual",
    )
    ax0.plot(
        df_pred.index,
        df_pred["price"],
        linewidth=0.3,
        c="tab:orange",
        alpha=0.9,
        label="Predicted",
    )
    ax0.legend()

    ax1.axhline(0, linewidth=0.7, color="black")
    ax1.grid(axis="x", linestyle="--")
    ax1.set_ylabel("Residual Price (€/MWHr)", fontsize=15)

    ax1.plot(
        resid_price.index, resid_price["price"], linewidth=0.5, alpha=0.9, c="tab:red"
    )
    ax1.fill_between(
        resid_price.index, resid_price["price"], color="tab:red", alpha=0.2
    )

    ax1.tick_params(axis="x", labelsize=15)

    return fig, axes


def plot_sqresid(df: pd.DataFrame, lag_max: int = 48):
    """Plot the ACF / PACF of the squared residuals.

    Args:
        df (pd.DataFrame): DataFrame containing the residual column `price`.
        lag_max (int, optional): The max lag to test. Defaults to 48.

    Returns:
        fig (plt.Fig)
        axes (np.array)
    """

    sq_resids = df["price"].to_numpy() ** 2

    fig, axes = plt.subplots(2, sharex=False, figsize=(12, 8), tight_layout=True)
    ax0, ax1 = axes[0], axes[1]

    for x in [t for t in range(lag_max + 1) if t % 24 != 0]:
        ax0.axvline(x, color="gray", linestyle="--", linewidth=0.5, alpha=0.9)
    for x in [t for t in range(0, lag_max + 1, 24) if t != 0]:
        ax0.axvline(x, color="black", linewidth=1, linestyle="--")
    plot_acf(
        sq_resids,
        ax=ax0,
        lags=[_ for _ in range(1, lag_max + 1)],
        linewidth=0,
        vlines_kwargs={"linewidth": 2},
        markersize=6,
    )
    ax0.axhline(0, color="black", linewidth=0.7)
    ax0.set_xticks([1] + [_ for _ in range(2, lag_max + 1, 2)])
    ax0.set_title("Autocorrelation", fontsize=15)

    for x in [t for t in range(lag_max + 1) if t % 24 != 0]:
        ax1.axvline(x, color="gray", linestyle="--", linewidth=0.5, alpha=0.9)
    for x in [t for t in range(0, lag_max + 1, 24) if t != 0]:
        ax1.axvline(x, color="black", linewidth=1, linestyle="--")
    plot_pacf(
        sq_resids,
        ax=ax1,
        lags=[_ for _ in range(1, lag_max + 1)],
        linewidth=0,
        vlines_kwargs={"linewidth": 2},
        markersize=6,
    )
    ax1.axhline(0, color="black", linewidth=0.7)
    ax1.set_xticks([1] + [_ for _ in range(2, lag_max + 1, 2)])
    ax1.set_xlabel("Lag (Hours)", fontsize=15)
    ax1.set_title("Partial Autocorrelation", fontsize=15)

    return fig, axes
