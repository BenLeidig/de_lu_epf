from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from de_lu_epf.plots.predictions import (
    plot_predictions,
    plot_residuals,
    plot_residuals_barplot,
    plot_residuals_raincloud,
    plot_residuals_violinplot,
)
from de_lu_epf.plots.utils import update_matplotlib_params

if __name__ == "__main__":
    update_matplotlib_params()

    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    PREDS_DIR = BASE_DIR / "data/predictions"
    FIGS_DIR = BASE_DIR / "reports/figures"

    train_val_start = pd.to_datetime("2019-02-01 00:00:00", utc=True)
    train_val_end = pd.to_datetime("2023-12-31 23:00:00", utc=True)

    test_start = pd.to_datetime("2024-02-01 00:00:00", utc=True)
    test_end = pd.to_datetime("2024-12-31 23:00:00", utc=True)

    df_actual = pd.read_parquet(DATA_DIR / "interim/merged.parquet")
    df_train_val_actual = df_actual[
        (df_actual.index >= train_val_start) & (df_actual.index <= train_val_end)
    ]
    df_test_actual = df_actual[
        (df_actual.index >= test_start) & (df_actual.index <= test_end)
    ]

    train_val_actual = df_train_val_actual["price"]
    test_actual = df_test_actual["price"]

    train_val_preds = {}
    for f in PREDS_DIR.rglob("*_train_val_pred.parquet"):
        model_name = f.name.split("_")[0].upper()
        df_pred = pd.read_parquet(f)
        df_pred = df_pred[
            (df_pred.index >= train_val_start) & (df_pred.index <= train_val_end)
        ]  # type: ignore
        train_val_preds[model_name] = df_pred["price"]

    test_preds = {}
    for f in PREDS_DIR.rglob("*_test_pred.parquet"):
        model_name = f.name.split("_")[0].upper()
        df_pred = pd.read_parquet(f)
        df_pred = df_pred[(df_pred.index >= test_start) & (df_pred.index <= test_end)]  # type: ignore
        test_preds[model_name] = df_pred["price"]

    train_val_idx = df_train_val_actual.index
    test_idx = df_test_actual.index

    #############################################################################################################################

    # PLOTTING TRAIN_VAL PREDS
    fig, _ = plot_predictions(
        index=train_val_idx,
        actual_series_dict={"Actual": train_val_actual},
        pred_series_dict=train_val_preds,
        suptitle="Predicted Price on the Train and Validation Sets",
    )
    fig.savefig(FIGS_DIR / "predictions/train_val_predictions.svg")

    # TEST SET PREDICTIONS
    months = pd.date_range("2024-02-01", "2024-12-01", freq="MS", tz="utc")

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 4)

    handles, labels = None, None

    for i, month_start in enumerate(months):
        row, col = divmod(i, 4)
        ax = fig.add_subplot(gs[row, col])

        month_end = month_start + pd.offsets.MonthEnd(1)
        mask = (test_idx >= month_start) & (test_idx <= month_end)

        idx_m = test_idx[mask]
        actual_m = test_actual[mask]
        preds_m = {model: preds[mask] for model, preds in test_preds.items()}

        plot_predictions(
            index=idx_m,
            actual_series_dict={"Actual": actual_m},
            pred_series_dict=preds_m,
            ax=ax,
        )

        ax.set_title(month_start.strftime("%b"))

        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

        ax.legend().remove()
    legend_ax = fig.add_subplot(gs[2, 3])
    legend_ax.axis("off")

    legend_ax.legend(
        handles,
        labels,
        loc="center",
        fontsize=10,
        frameon=False,
        ncol=1,
    )

    fig.suptitle("Monthly Test Set Predictions (2024)")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "predictions/monthly_test_predictions.svg")

    #############################################################################################################################

    # PLOTTING TRAIN_VAL RESIDUALS
    fig, _ = plot_residuals(
        index=train_val_idx,
        actual_series=train_val_actual,
        pred_series_dict=train_val_preds,
        suptitle="Residuals on the Train and Validation Sets",
    )
    fig.savefig(FIGS_DIR / "predictions/train_val_residuals.svg")

    # PLOTTING TEST RESIDUALS
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 4)

    handles, labels = None, None

    for i, month_start in enumerate(months):
        row, col = divmod(i, 4)
        ax = fig.add_subplot(gs[row, col])

        month_end = month_start + pd.offsets.MonthEnd(1)

        mask = (test_idx >= month_start) & (test_idx <= month_end)

        idx_m = test_idx[mask]
        actual_m = test_actual[mask]
        preds_m = {model: preds[mask] for model, preds in test_preds.items()}

        plot_residuals(
            index=idx_m,
            actual_series=actual_m,
            pred_series_dict=preds_m,
            ax=ax,
        )

        ax.set_title(month_start.strftime("%b"))

        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

        ax.legend().remove()

    legend_ax = fig.add_subplot(gs[2, 3])
    legend_ax.axis("off")

    legend_ax.legend(
        handles,
        labels,
        loc="center",
        fontsize=10,
        frameon=False,
        ncol=1,
    )

    fig.suptitle("Monthly Test Set Residuals (2024)")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "predictions/monthly_test_residuals.svg")

    #############################################################################################################################

    # VIOLINPLOT TRAIN_VAL RESIDUALS
    fig, _ = plot_residuals_violinplot(
        actual_series=train_val_actual,
        pred_series_dict=train_val_preds,
        suptitle="Distribution of Train and Validation Set Predictions",
    )
    fig.savefig(FIGS_DIR / "predictions/train_val_residuals_violinplot.svg")

    # VIOLINPLOT TEST RESIDUALS
    fig, _ = plot_residuals_violinplot(
        actual_series=test_actual,
        pred_series_dict=test_preds,
        suptitle="Distribution of Test Set Predictions",
    )
    fig.savefig(FIGS_DIR / "predictions/test_residuals_violinplot.svg")

    metrics = [mean_absolute_error, r2_score, root_mean_squared_error]

    #############################################################################################################################

    # TRAIN_VAL METRIC BARPLOTS
    for metric in metrics:
        metric_name = metric.__name__.split("_")
        metric_name = [m[0].upper() + m[1:] for m in metric_name]
        metric_name = " ".join(metric_name)
        fig, _ = plot_residuals_barplot(
            actual_series=train_val_actual,
            pred_series_dict=train_val_preds,
            metric=metric,
            suptitle=f"{metric_name} Barplot for Train and Validation Set Predictions",
        )
        fig.savefig(FIGS_DIR / f"predictions/train_val_{metric.__name__}.svg")

    # TEST METRIC BARPLOTS
    for metric in metrics:
        metric_name = metric.__name__.split("_")
        metric_name = [m[0].upper() + m[1:] for m in metric_name]
        metric_name = " ".join(metric_name)
        fig, _ = plot_residuals_barplot(
            actual_series=test_actual,
            pred_series_dict=test_preds,
            metric=metric,
            suptitle=f"{metric_name} Barplot for Test Set Predictions",
        )
        fig.savefig(FIGS_DIR / f"predictions/test_{metric.__name__}.svg")

    #############################################################################################################################

    # RAINCLOUDPLOT TRAIN_VAL RESIDUALS
    fig, _ = plot_residuals_raincloud(
        actual_series=train_val_actual,
        pred_series_dict=train_val_preds,
        suptitle="Distribution of Train and Validation Set Predictions",
    )
    fig.savefig(FIGS_DIR / "predictions/train_val_residuals_raincloudplot.svg")

    # RAINCLOUDPLOT TEST RESIDUALS
    fig, _ = plot_residuals_raincloud(
        actual_series=test_actual,
        pred_series_dict=test_preds,
        suptitle="Distribution of Test Set Predictions",
    )
    fig.savefig(FIGS_DIR / "predictions/test_residuals_raincloudplot.svg")
