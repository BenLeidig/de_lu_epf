from pathlib import Path

import torch
import yaml
from joblib import dump

from src.models.hpo.tune_vtlm.utils import tune_vmd_tcn_lstm_mha

if __name__ == "__main__":
    # Set paths
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    cfg_path = BASE_DIR / "configs/models/hpo_config.yaml"
    studies_path = BASE_DIR / "studies"

    # Set configs
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["vmd-tcn-lstm-mha"]

    target_col = "imf_resid"  # Declare target signal
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Create study
    study = tune_vmd_tcn_lstm_mha(target_col=target_col, **cfg)
    dump(study, studies_path / f"{target_col}_tcn_lstm_mha.pkl")  # Save study
