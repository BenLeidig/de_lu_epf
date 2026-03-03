from joblib import dump
import pandas as pd
from model_eval import get_all_preds

HOME_PATH = '/u/bleidig2/'
SCRATCH_PATH = '/u/bleidig2/scratch/'

train_val_dt_index = pd.read_pickle(SCRATCH_PATH+'df_train_val.pkl').index
test_dt_index = pd.read_pickle(SCRATCH_PATH+'df_test.pkl').index

train_val_preds, train_val_preds_scaled, test_preds, test_preds_scaled = get_all_preds(
    study_dir=HOME_PATH,
    data_dir=SCRATCH_PATH,
    scaler_dir=HOME_PATH,
    max_epochs=100,
    accelerator='gpu'
)

## setting train_val dt index
train_val_dt_index = train_val_dt_index[-len(train_val_preds):]
train_val_preds = train_val_preds.set_index(train_val_dt_index)
train_val_preds_scaled = train_val_preds_scaled.set_index(train_val_dt_index)

## setting test dt index
test_dt_index = test_dt_index[-len(test_preds):]
test_preds = test_preds.set_index(test_dt_index)
test_preds_scaled = test_preds_scaled.set_index(test_dt_index)


dump(train_val_preds, 'df_train_val_pred_vtlm.pkl')
dump(train_val_preds_scaled, 'df_train_val_scaled_pred_vtlm.pkl')
dump(test_preds, 'df_test_pred_vtlm.pkl')
dump(test_preds_scaled, 'df_test_scaled_pred_vtlm.pkl')