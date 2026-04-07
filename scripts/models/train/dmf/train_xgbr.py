from pathlib import Path

from joblib import dump

from de_lu_epf.models.training import get_fitted_dmf

if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent.parent.parent.parent
    MODEL_DIR = BASE_DIR / "models/dmf/full"

    model_name = "xgbr"
    dmf = get_fitted_dmf(model_name=model_name)

    dump(dmf, MODEL_DIR / f"{model_name}.pkl")
