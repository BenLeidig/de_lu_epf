import os
from joblib import dump
from hpo import vmd_tcn_lstm_mha_hpo

NETID = os.environ['USER']
SCRATCH_PATH = f'/scratch/{NETID}/'

if __name__ == '__main__':
    study = vmd_tcn_lstm_mha_hpo(
        imf=5,
        data_dir=SCRATCH_PATH,
        optimize_kwargs={'n_trials':500}
    )
    dump(study, 'imf5_tcn_lstm_mha_study.pkl')