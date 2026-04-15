import de_lu_epf.models.architectures as arc
from de_lu_epf.models.training import get_best_ann_params, get_fitted_ann

if __name__ == "__main__":
    model_name = "lstm"
    model_class = arc.LSTM
    model_type = "ann"

    target_col = "price"

    batch_size, params = get_best_ann_params(
        target_col=target_col, model_name=model_name, model_type=model_type
    )  # target_col not actually needed here for ANN non-Hybrid models

    get_fitted_ann(
        model_class=model_class,
        params=params,
        target_col=target_col,
        model_type=model_type,
        model_name=model_name,
        seq_len=24 * 7 * 2,
        pred_len=24,
        stride=24,
        batch_size=batch_size,
        patience=5,
        max_epochs=100,
        random_state=0,
    )
