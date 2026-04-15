from pathlib import Path

import de_lu_epf.models.architectures as arc
from de_lu_epf.models.predicting import get_predictions_ann

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
    DATA_DIR = BASE_DIR / "data/predictions"

    model_name = "tcn_lstm_mha"
    model_class = arc.TCN_LSTM_MHA
    model_type = "ann"

    y_train_val_pred, y_test_pred = get_predictions_ann(
        model_name=model_name, model_class=model_class
    )

    y_train_val_pred.to_parquet(
        DATA_DIR / f"train_val/{model_type}/{model_name}_train_val_pred.parquet",
        index=True,
    )
    y_test_pred.to_parquet(
        DATA_DIR / f"test/{model_type}/{model_name}_test_pred.parquet", index=True
    )
