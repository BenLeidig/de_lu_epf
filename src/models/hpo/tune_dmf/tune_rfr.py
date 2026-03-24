import multiprocessing
import os
from pathlib import Path

import yaml
from joblib import dump
from sklearn.ensemble import RandomForestRegressor

from src.models.hpo.tune_dmf.utils import create_dmf_study

if __name__ == "__main__":
    # Set paths
    BASE_DIR = Path(__file__).parent.parent.parent.parent
    cfg_path = BASE_DIR / "configs/models/hpo_config.yaml"
    studies_path = BASE_DIR / "studies"

    # Set configs
    name = "rfr"
    with open(cfg_path) as f:
        search_space = yaml.safe_load(f)[name]

    # Set model class
    model_class = RandomForestRegressor

    # Get forecasting hour and get multithreading capacity
    hour = int(os.environ["SLURM_ARRAY_TASK_ID"])
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))

    # Create DMF study
    study = create_dmf_study(
        hour=hour, model_class=model_class, search_space=search_space, n_jobs=n_jobs
    )

    # Save study
    dump(study, studies_path / f"hour{hour}_{name}.pkl")
