from pathlib import Path

from de_lu_epf.models.predicting import get_predictions_dmf

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
    DATA_DIR = BASE_DIR / "data/predictions"

    model_name = "lgbmr"
    y_train_val_pred, y_test_pred = get_predictions_dmf(model_name)

    y_train_val_pred.to_parquet(
        DATA_DIR / f"train_val/dmf/{model_name}_train_val_pred.parquet", index=True
    )
    y_test_pred.to_parquet(
        DATA_DIR / f"test/dmf/{model_name}_test_pred.parquet", index=True
    )
