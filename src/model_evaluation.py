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
    numerator = np.sqrt((1/n) * (np.sum(y_pred-y_true)**2))
    denominator = np.sqrt((1/n) * np.sum(y_true**2)) + np.sqrt((1/n) * np.sum(y_pred**2))
    tic = numerator / denominator
    return tic

def absolute_percentage_bias(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    apb = np.abs(np.sum(y_true-y_pred) / np.sum(y_true))
    return apb

def evaluate_model(model, X, y_true) -> pd.DataFrame:

    '''
    model   :   A pre-trained model object.
    X       :   Feature matrix. Should be in the format your model object requires.
    y_true  :   True target array for prediction evaluation.
    '''

    if hasattr(model, 'predict'):
        y_pred = model.predict(X)
    elif callable(model):
        y_pred = model(X)
    else:
        raise TypeError('Model must be callable or have a .predict() method.')
    
    abbr = [
        'r2', 'wi', 'ns', 'lm', 'kge',
        'rmse', 'mae', 'nrmse', 'rmae', 'smape', 'tic', 'apb'
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