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

## NOTE: here we are programmatically generating some exploratory
## visualizations for each dataset (train / validation / test).
## Please see src/de_lu_epf/plots/ for more code details.
##
## Two distinct phases, not to be mixed:
##   1) Model comparison ("who's best?") - driven entirely by "train"/"val"
##      predictions (every candidate model's own held-out validation split,
##      see scripts/models/predict/**/*.py). This is the only place a
##      cross-architecture ranking bar chart belongs.
##   2) Final reporting - once scripts/models/select_final_model.py has
##      picked the single overall winner by validation performance, retrained
##      it on train_val, and evaluated it once on test, "train_val"/"test"
##      hold only that one model's predictions. The plots for those splits
##      are single-model diagnostics (time series, residuals), not a
##      cross-model ranking - there is deliberately no ranking bar chart
##      here, since ranking on test would defeat the point of only ever
##      evaluating the already-selected model on it.

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    SAVE_DIR = BASE_DIR / "reports/figures/predictions"

    df_actual = pd.read_parquet(DATA_DIR / "processed/processed.parquet")

    # --- Phase 1: compare all candidates by validation performance ---
    for set in ["train", "val"]:
        df_preds = get_all_set_preds(set=set)
        actual_series = df_actual.loc[df_preds.index, "price"]

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

    # --- Phase 2: diagnostics for the single, already-selected final model ---
    for set, set_name in zip(["train_val", "test"], ["Train+Val", "Test"]):
        df_preds = get_all_set_preds(set=set)

        actual_series = df_actual.loc[df_preds.index, "price"]

        fig = plot_predictions_interactive(
            actual_series=actual_series,
            pred_df=df_preds,
            title=f"Interactive {set_name} Set Predictions (Final Model)",
        )
        fig.write_html(SAVE_DIR / f"interactive_{set}_preds.html")

        fig = plot_residuals_interactive(
            actual_series=actual_series,
            pred_df=df_preds,
            title=f"Interactive {set_name} Set Residuals (Final Model)",
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
