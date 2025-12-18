import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

def willmotts_index(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    wi = 1 - (np.sum((y_true-y_pred)**2) / np.sum((np.abs(y_pred-np.mean(y_true))+(np.abs(y_true-np.mean(y_pred))))**2))
    return wi

def nash_sutcliffe_efficiency(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ns = 1 - np.sum((y_true-y_pred)**2) / np.sum((y_true-np.mean(y_true))**2)
    return ns

def legates_mccabes_index(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    lm = 1 - np.sum(np.abs(y_pred-y_true)) / np.sum(np.abs(y_true-np.mean(y_true)))
    return lm

def kling_gupta_efficiency(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cv_true = np.std(y_true) / np.mean(y_true)
    cv_pred = np.std(y_pred) / np.mean(y_pred)
    r = np.sum((y_true - y_true.mean()) * (y_pred - y_pred.mean())) / np.sqrt(np.sum((y_true - y_true.mean())**2) * np.sum((y_pred - y_pred.mean())**2))
    kge = 1 - np.sqrt((r-1)**2 + (np.mean(y_pred)/np.mean(y_true) - 1)**2 + (cv_pred/cv_true)**2)
    return kge

def normalized_root_mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    nrmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred) / np.mean(y_true)
    return nrmse

def relative_mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmae = mean_absolute_error(y_true=y_true, y_pred=y_pred) / np.mean(y_true)
    return rmae

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    smape = (1/len(y_true)) * np.sum(np.abs(y_true-y_pred) / ((np.abs(y_true) + np.abs(y_pred))/2))
    return smape

def theils_inequality_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    numerator = np.sqrt((1/n) * (np.sum((y_pred-y_true)**2)))
    denominator = np.sqrt((1/n) * np.sum(y_true**2)) + np.sqrt((1/n) * np.sum(y_pred**2))
    tic = numerator / denominator
    return tic

def absolute_percentage_bias(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    apb = np.abs(np.sum(y_true-y_pred) / np.sum(y_true))
    return apb

def cumscore(y_true, y_pred, metric):
    cumscores = []
    for i in range(1, len(y_true)+1):
        score = metric(y_true=y_true[:i], y_pred=y_pred[:i])
        cumscores.append(score)
    return cumscores

def evaluate_model(y_true, y_pred) -> pd.DataFrame:

    '''
    model   :   A pre-trained model object.
    X       :   Feature matrix. Should be in the format your model object requires.
    y_true  :   True target array for prediction evaluation.
    '''
    
    abbr = [
        'r2', 'wi', 'ns', 'lm', 'kge',
        'rmse', 'mae', 'nrmse', 'rmape', 'smape', 'tic', 'apb'
    ]
    prop = ['variance']*5 + ['bias']*7
    metrics = [
        r2_score, willmotts_index, nash_sutcliffe_efficiency, legates_mccabes_index, kling_gupta_efficiency,
        root_mean_squared_error, mean_absolute_error, normalized_root_mean_squared_error, relative_mean_absolute_percentage_error, symmetric_mean_absolute_percentage_error, theils_inequality_coefficient, absolute_percentage_bias
    ]
    results = [
        metric(y_true=y_true, y_pred=y_pred) for metric in metrics
    ]

    df = pd.DataFrame(
        data=list(zip(prop, abbr, results)),
        columns=['property', 'metric', 'score']
    )

    return df

def cum_rmse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    cum_sum_sq = np.cumsum((y_true - y_pred)**2)
    counts = np.arange(1, len(y_true)+1)
    return np.sqrt(cum_sum_sq / counts)

def cum_mae(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    cum_sum_abs = np.cumsum(np.abs(y_true - y_pred))
    counts = np.arange(1, len(y_true)+1)
    return cum_sum_abs / counts

def cum_nrmse(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    rmse = cum_rmse(y_true, y_pred)
    global_mean = np.mean(y_true)
    if global_mean == 0:
        return np.full_like(rmse, np.nan, dtype=float)
    return rmse / global_mean

def cum_rmae(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    mae = cum_mae(y_true, y_pred)
    global_mean = np.mean(y_true)
    if global_mean == 0:
        return np.full_like(mae, np.nan, dtype=float)
    return mae / global_mean

def cum_smape(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    abs_diff = np.abs(y_true - y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    ratio = np.divide(abs_diff, denom, out=np.zeros_like(abs_diff, dtype=float), where=denom != 0)
    cum_sum = np.cumsum(ratio)
    counts = np.arange(1, len(y_true)+1)
    return cum_sum / counts

def cum_apb(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    global_sum = np.sum(y_true)
    cum_num = np.cumsum(y_true - y_pred)
    if global_sum == 0:
        return np.full_like(cum_num, np.nan, dtype=float)
    return np.abs(cum_num / global_sum)

def cum_tic(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n_total = len(y_true)
    numerator = np.sqrt(np.cumsum((y_pred - y_true)**2) / np.arange(1, n_total+1))
    global_den1 = np.sqrt(np.sum(y_true**2) / n_total)
    global_den2 = np.sqrt(np.sum(y_pred**2) / n_total)
    denominator = global_den1 + global_den2
    if denominator == 0:
        return np.full_like(numerator, np.nan, dtype=float)
    return numerator / denominator