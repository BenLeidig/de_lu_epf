from pathlib import Path

from joblib import load
from yaml import safe_dump

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
    STUDY_DIR = BASE_DIR / "studies/hybrid"
    CFG_PATH = BASE_DIR / "configs/models/hybrid_hyperparams_config.yaml"

    imf_dict = {}
    imf_mods = ["vtlm"]

    for mod_name in imf_mods:
        imf_dict[mod_name] = {}

        for mod_file in STUDY_DIR.rglob(f"*{mod_name}.pkl"):
            imf = mod_file.name.rsplit("_", 1)[0]
            best_params = load(mod_file).best_params
            imf_dict[mod_name][imf] = best_params

    with open(CFG_PATH, "w") as f:
        safe_dump(imf_dict, f, sort_keys=False)
