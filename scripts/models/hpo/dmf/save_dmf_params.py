from pathlib import Path

from joblib import dump, load
from yaml import safe_dump

from de_lu_epf.models.training import get_fitted_dmf

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
    STUDY_DIR = BASE_DIR / "studies/dmf"
    CFG_PATH = BASE_DIR / "configs/models/dmf_hyperparams_config.yaml"
    CANDIDATES_DIR = BASE_DIR / "models/dmf/candidates"

    dmf_dict = {}
    dmf_mods = ["en", "lgbmr", "rfr", "svr", "xgbr"]

    for mod_name in dmf_mods:
        dmf_dict[mod_name] = {}

        for mod_file in STUDY_DIR.rglob(f"*{mod_name}.pkl"):
            hour = int(mod_file.name.split("_")[0].replace("hour", ""))
            best_params = load(mod_file).best_params
            dmf_dict[mod_name][hour] = best_params

    with open(CFG_PATH, "w") as f:
        safe_dump(dmf_dict, f, sort_keys=False)

    # DMF's HPO evaluates each hyperparameter config via 5-fold time-series
    # cross-validation (see dmf_tuning.py) - unlike the neural architectures,
    # no single fitted model naturally falls out of a trial to preserve
    # directly. Once the winning hyperparameters are known (just written
    # above), fit each model once on the full "train" split and save it as
    # this family's "candidate" - this is cheap for classical ML (seconds,
    # not hours), so doing it right here removes the need for a separate
    # train_*.py step to produce it. "lr" has no tunable hyperparameters
    # (see get_fitted_dmf), but still needs its own candidate fit.
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    for mod_name in ["lr"] + dmf_mods:
        dmf = get_fitted_dmf(model_name=mod_name, final=False)
        dump(dmf, CANDIDATES_DIR / f"{mod_name}.pkl")
