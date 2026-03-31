from pathlib import Path

from joblib import load
from yaml import safe_dump

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
    STUDY_DIR = BASE_DIR / "studies/dmf"
    CFG_PATH = BASE_DIR / "configs/models/dmf_hyperparams_config.yaml"

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
