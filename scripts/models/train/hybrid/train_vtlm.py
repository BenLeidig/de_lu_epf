import os
from pathlib import Path

from joblib import dump

from de_lu_epf.models.architectures import TCN_LSTM_MHA
from de_lu_epf.models.training import get_best_vtlm_params, get_fitted_ann

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
    MODEL_DIR = BASE_DIR / "models/hybrid/full"
    DATA_DIR = BASE_DIR / "data/processed/hybrid"

    model_name = "vtlm"

    i = int(os.environ["SLURM_ARRAY_TASK_ID"])
    targets = ["imf1", "imf2", "imf3", "imf4", "imf5", "imf_resid"]
    target_col = targets[i]
    epochs = 35

    batch_size, params = get_best_vtlm_params(target_col=target_col)

    dmf = get_fitted_ann(
        target_col=target_col,
        batch_size=batch_size,
        params=params,
        data_dir=DATA_DIR,
        model_class=TCN_LSTM_MHA,
        max_epochs=epochs,
    )

    dump(dmf, MODEL_DIR / f"vtlm/{target_col}_{model_name}.pkl")
