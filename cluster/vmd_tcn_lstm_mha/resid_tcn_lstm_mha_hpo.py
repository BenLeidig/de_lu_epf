import os
from joblib import dump
from hpo import vmd_tcn_lstm_mha_hpo

NETID = os.environ['USER']
SCRATCH_PATH = f'/scratch/{NETID}/'

if __name__ == '__main__':
    study = vmd_tcn_lstm_mha_hpo(
        imf=6,
        data_dir=SCRATCH_PATH,
        optimize_kwargs={'n_trials':500}
    )
    dump(study, 'resid_tcn_lstm_mha_study.pkl')