import os
from joblib import dump
import numpy as np
import pandas as pd
from sktime.transformations.series.vmd import VmdTransformer

NETID = os.environ['USER']
SCRATCH_PATH = f'/scratch/{NETID}'
DATA_PATH = os.path.join(os.environ['HOME'], 'df_train_val.pkl')
DUMP_PATH = os.path.join(SCRATCH_PATH, 'vmd_fc.pkl')

price = pd.read_pickle(DATA_PATH)['price'].to_numpy()

def fc(x:np.ndarray):
    X = np.fft.rfft(x)
    power = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(len(x))
    return np.sum(freqs*power) / np.sum(power)

def vmd_transform(K:int, y:np.ndarray):
    vmd = VmdTransformer(K=K, alpha=4_000)
    imfs = vmd.fit_transform(y)
    fc_dict = {i: fc(imfs[:, i]) for i in range(imfs.shape[1])}
    return K, fc_dict

K_dict = {}
for K in range(2, 21):
    K_dict[K] = vmd_transform(K=K, y=price)[1]
    print(f'{K} complete.')

dump(K_dict, DUMP_PATH)