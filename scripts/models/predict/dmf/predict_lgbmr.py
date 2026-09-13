from pathlib import Path

from de_lu_epf.models.predicting import get_predictions_dmf

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
    DATA_DIR = BASE_DIR / "data/predictions"

    model_name = "lgbmr"
    # Comparison-mode predictions: this "candidate" model was fit on
    # train, held out from val, so its val-split predictions can be
    # compared against other candidates' by validation performance.
    # Only the single overall winner gets a train_val refit and a
    # one-time test evaluation - see select_final_model.py.
    y_train_pred, y_val_pred = get_predictions_dmf(
        model_name=model_name,
        train_split="train",
        test_split="val",
    )

    y_train_pred.to_parquet(
        DATA_DIR / f"train/dmf/{model_name}_train_pred.parquet", index=True
    )
    y_val_pred.to_parquet(
        DATA_DIR / f"val/dmf/{model_name}_val_pred.parquet", index=True
    )
