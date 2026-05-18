from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from de_lu_epf.models.predicting import get_all_set_preds
from de_lu_epf.plots.predictions import (
    get_metric_name,
    plot_metric_barplot,
    plot_month_preds,
    plot_predictions_interactive,
    plot_residual_violinplot,
    plot_residuals_interactive,
)

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    SAVE_DIR = BASE_DIR / "reports/figures/predictions"

    df_actual = pd.read_parquet(DATA_DIR / "processed/processed.parquet")
    for set, set_name in zip(["train_val", "test"], ["Train", "Test"]):
        df_preds = get_all_set_preds(set=set)

        actual_series = df_actual.loc[df_preds.index, "price"]

        fig = plot_predictions_interactive(
            actual_series=actual_series,
            pred_df=df_preds,
            title=f"Interactive {set_name} Set Predictions",
        )
        fig.write_html(SAVE_DIR / f"interactive_{set}_preds.html")

        fig = plot_residuals_interactive(
            actual_series=actual_series,
            pred_df=df_preds,
            title=f"Interactive {set_name} Set Residuals",
        )
        fig.write_html(SAVE_DIR / f"interactive_{set}_resids.html")

        fig, ax = plot_residual_violinplot(
            actual_series=actual_series,
            pred_df=df_preds,
            title=None,
        )
        fig.savefig(SAVE_DIR / f"{set}_residual_distribution.svg")

        if set == "test":
            fig, ax = plot_month_preds(
                month=2, year=2024, actual_series=actual_series, pred_df=df_preds
            )
            fig.savefig(SAVE_DIR / f"{set}_feb_preds.svg")

            fig, ax = plot_month_preds(
                month=12, year=2024, actual_series=actual_series, pred_df=df_preds
            )
            fig.savefig(SAVE_DIR / f"{set}_dec_preds.svg")

        for metric, ascending in zip(
            [mean_absolute_error, r2_score, root_mean_squared_error],
            [True, False, True],
        ):
            metric_name = get_metric_name(metric=metric)

            fig, _ = plot_metric_barplot(
                actual_series=actual_series,
                pred_df=df_preds,
                metric=metric,
                ascending=ascending,
                title=None,
            )
            fig.savefig(SAVE_DIR / f"{set}_{metric.__name__}.svg")
