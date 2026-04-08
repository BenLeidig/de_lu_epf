from pathlib import Path

import torch
import yaml
from joblib import dump

from de_lu_epf.models.hpo.ann_tuning import tune_tcn_lstm_mha

if __name__ == "__main__":
    model_type = "ann"
    model_name = "tcn_lstm_mha"
    tuner = tune_tcn_lstm_mha

    # Set paths
    BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
    cfg_path = BASE_DIR / "configs/models/hpo_config.yaml"
    studies_path = BASE_DIR / f"studies/{model_type}"

    # Set configs
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["tcn-lstm-mha"]

    # Settings
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Create study
    study = tuner(target_col="price", model_type=model_type, **cfg)

    # Save study
    dump(study, studies_path / f"{model_name}.pkl")
