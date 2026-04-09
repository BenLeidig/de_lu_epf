import os
from pathlib import Path

import torch
import yaml
from joblib import dump

from de_lu_epf.models.hpo.ann_tuning import tune_lstm

if __name__ == "__main__":
    model_type = "hybrid"
    model_name = "vl"
    tuner = tune_lstm

    # Set paths
    BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
    cfg_path = BASE_DIR / "configs/models/hpo_config.yaml"
    studies_path = BASE_DIR / f"studies/{model_type}/{model_name}"

    # Set configs
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["lstm"]

    # Get IMF
    i = int(os.environ["SLURM_ARRAY_TASK_ID"])
    targets = ["imf1", "imf2", "imf3", "imf4", "imf5", "imf_resid"]
    target_col = targets[i]

    # Settings
    torch.multiprocessing.set_start_method("spawn", force=True)

    # Create study
    study = tuner(target_col=target_col, model_type=model_type, **cfg)

    # Save study
    dump(study, studies_path / f"{target_col}_{model_name}.pkl")
