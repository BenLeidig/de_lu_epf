import os
from joblib import dump
import numpy as np
import pandas as pd
from sktime.transformations.series.vmd import VmdTransformer

NETID = os.environ['USER']
SCRATCH_PATH = f'/scratch/{NETID}'
DATA_PATH = os.path.join(os.environ['HOME'], 'df_train_val.pkl')
DUMP_PATH = os.path.join(SCRATCH_PATH, 'vmd_corr.pkl')

price = pd.read_pickle(DATA_PATH)['price'].to_numpy()

def vmd_transform(K:int, y:np.ndarray):
    vmd = VmdTransformer(K=K, alpha=4_000)
    imfs = vmd.fit_transform(y)
    corr_dict = {i: np.abs(np.corrcoef(y, imfs[:, i])[0, 1]) for i in range(imfs.shape[1])}
    return K, corr_dict

K_dict = {}
for K in range(2, 21):
    K_dict[K] = vmd_transform(K=K, y=price)[1]
    print(f'{K} complete.')

dump(K_dict, DUMP_PATH)