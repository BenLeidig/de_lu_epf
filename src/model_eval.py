from joblib import load

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

from data_setup import EPFDataset, EPFDataModule
from model_setup import TCN_LSTM_MHA


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


def get_params(best_params:dict):
    batch_size = best_params.pop('batch_size')
    hidden_sizes = [best_params.pop('hidden_size0'), best_params.pop('hidden_size1'), best_params.pop('hidden_size2')]
    lstm_dropouts = [best_params.pop('lstm_dropout0'), best_params.pop('lstm_dropout1')]
    channel_sizes = []; channel_size_keys = list(best_params.keys())
    for s in channel_size_keys:
        if 'channel_size_' in s:
            channel_sizes.append(best_params.pop(s))
    params = best_params.copy()
    params['hidden_sizes'] = hidden_sizes
    params['lstm_dropouts'] = lstm_dropouts
    params['channel_sizes'] = channel_sizes
    return batch_size, params


def get_fitted(
        imf:int,
        best_params:dict,
        data_dir:str,
        max_epochs:int=200,
        patience:int=10,
        accelerator='auto'
):
    
    batch_size, params = get_params(best_params=best_params)

    datamodule = EPFDataModule(
        X_train_path=data_dir+'df_train_scaled.pkl',
        X_val_path=data_dir+'df_val_scaled.pkl',
        y_train_path=data_dir+'df_train_val_imf_scaled.pkl',
        y_val_path=data_dir+'df_val_imf_scaled.pkl',
        imf=imf,
        batch_size=batch_size
    )

    model = TCN_LSTM_MHA(input_size=8, **params)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=pl.callbacks.EarlyStopping(monitor='val_loss', patience=patience),
        accelerator=accelerator,
        logger=False,
        enable_checkpointing=False
    )

    trainer.fit(model, datamodule=datamodule)

    return batch_size, model, trainer


def get_preds(
        imf:int,
        study_dir:str,
        data_dir:str,
        max_epochs:int=100,
        patience:int=5,
        accelerator='auto'
):
    
    imf_str = 'resid' if imf==6 else 'imf'+str(imf)
    best_params = load(study_dir+imf_str+'_tcn_lstm_mha_study.pkl').best_params

    batch_size, model, trainer = get_fitted(
        imf=imf,
        best_params=best_params,
        data_dir=data_dir,
        max_epochs=max_epochs,
        patience=patience,
        accelerator=accelerator
    )

    X_test = torch.tensor(
        pd.read_pickle(data_dir+'df_test_scaled.pkl').to_numpy(),
        dtype=torch.float32
    )
    y_test = torch.tensor(
        pd.read_pickle(data_dir+'df_test_imf_scaled.pkl').to_numpy()[:, imf-1],
        dtype=torch.float32
    )
    test_dataset = EPFDataset(X=X_test, y=y_test)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True
    )
    preds = trainer.predict(model, test_loader)
    flat_list = [item for sublist in preds for inner in sublist for item in inner]
    y_test_scaled_pred = np.array(flat_list)

    return y_test_scaled_pred


def get_all_preds(
        study_dir:str,
        data_dir:str,
        scaler_dir:str,
        max_epochs:int=100,
        pateince:int=5,
        accelerator='auto'
):
    scaler = load(scaler_dir+'ss_df_train_imf.pkl')
    preds_scaled = []
    for imf in range(1, 7):
        preds_scaled.append(get_preds(
            imf=imf,
            study_dir=study_dir,
            data_dir=data_dir,
            max_epochs=max_epochs,
            patience=pateince,
            accelerator=accelerator
        ))
    preds_scaled = pd.DataFrame(
        data=np.array([_ for _ in preds_scaled]).T,
        columns=['imf'+str(i) for i in range(1, 6)]+['resid']
    )
    preds = pd.DataFrame(
        data=scaler.inverse_transform(preds_scaled),
        columns=preds_scaled.columns
    )
    return preds, preds_scaled