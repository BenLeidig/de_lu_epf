from pathlib import Path

from joblib import load
from yaml import safe_dump

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
    STUDY_DIR = BASE_DIR / "studies/ann"
    CFG_PATH = BASE_DIR / "configs/models/ann_hyperparams_config.yaml"

    d = {}
    mods = ["lstm_mha", "lstm", "tcn_lstm_mha", "tcn_lstm", "tcn_mha", "tcn"]

    for mod_name in mods:
        mod_key = mod_name.replace("_", "-")
        d[mod_key] = {}

        for mod_file in STUDY_DIR.rglob(f"{mod_name}.pkl"):
            best_params = load(mod_file).best_params
            d[mod_key] = best_params

    with open(CFG_PATH, "w") as f:
        safe_dump(d, f, sort_keys=False)
