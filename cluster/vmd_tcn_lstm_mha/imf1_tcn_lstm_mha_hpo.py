import os
from joblib import dump
import torch
from hpo import vmd_tcn_lstm_mha_hpo

NETID = os.environ['USER']
SCRATCH_PATH = f'/scratch/{NETID}/'

imf = 1

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    study = vmd_tcn_lstm_mha_hpo(
        imf=imf,
        data_dir=SCRATCH_PATH,
        patience=15,
        max_epochs=200,
        reduction_factor=3,
        optimize_kwargs={'n_trials':1_000},
        accelerator='gpu'
    )
    dump(study, f'imf{imf}_tcn_lstm_mha_study.pkl')