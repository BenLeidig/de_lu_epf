import os
from pathlib import Path

import torch
import yaml
from joblib import dump

from models.hpo.hybrid_tuning import tune_vmd_tcn_lstm_mha

if __name__ == "__main__":
    # Set paths
    BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
    cfg_path = BASE_DIR / "configs/models/hpo_config.yaml"
    studies_path = BASE_DIR / "studies/hybrid/vtlm"

    # Set configs
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["vmd-tcn-lstm-mha"]

    # Get IMF
    i = int(os.environ["SLURM_ARRAY_TASK_ID"])
    targets = ["imf1", "imf2", "imf3", "imf4", "imf5", "imf_resid"]
    target_col = targets[i]

    # Settings
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Create study
    study = tune_vmd_tcn_lstm_mha(target_col=target_col, **cfg)

    # Save study
    dump(study, studies_path / f"{target_col}_tcn_lstm_mha.pkl")
