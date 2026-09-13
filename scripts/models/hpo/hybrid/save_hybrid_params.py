import shutil
from pathlib import Path

from joblib import load
from yaml import safe_dump

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
    STUDY_DIR = BASE_DIR / "studies/hybrid"
    CFG_PATH = BASE_DIR / "configs/models/hybrid_hyperparams_config.yaml"
    CANDIDATES_DIR = BASE_DIR / "models/hybrid/candidates"

    imf_dict = {}
    imf_mods = ["vl", "vlm", "vt", "vtl", "vtlm", "vtm"]

    for mod_name in imf_mods:
        imf_dict[mod_name] = {}
        mod_candidates_dir = CANDIDATES_DIR / mod_name
        mod_candidates_dir.mkdir(parents=True, exist_ok=True)

        for mod_file in STUDY_DIR.rglob(f"*{mod_name}.pkl"):
            imf = mod_file.name.rsplit("_", 1)[0]
            best_params = load(mod_file).best_params
            imf_dict[mod_name][imf] = best_params

            # Publish HPO's own preserved best-trial checkpoint for this IMF
            # (see ann_tuning.py's _fit_and_checkpoint / tune_*.py's
            # checkpoint_path) as this IMF's "candidate" model - no separate
            # train_*.py refit needed to reproduce it.
            ckpt_file = mod_file.parent / f"{imf}_{mod_name}_best.ckpt"
            if ckpt_file.exists():
                shutil.copyfile(ckpt_file, mod_candidates_dir / f"{imf}_{mod_name}.ckpt")

    with open(CFG_PATH, "w") as f:
        safe_dump(imf_dict, f, sort_keys=False)
