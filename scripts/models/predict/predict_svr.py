from pathlib import Path

from src.models.predicting import get_predictions_dmf

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    DATA_DIR = BASE_DIR / "data/predictions/dmf"

    model_name = "svr"
    y_test_pred = get_predictions_dmf(model_name)
    y_test_pred.to_parquet(DATA_DIR / "test_pred.parquet", index=True)
