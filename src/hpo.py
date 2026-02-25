import numpy as np

import lightning.pytorch as pl

import optuna

from data_setup import EPFDataModule
from model_setup import TCN_LSTM_MHA


def vmd_tcn_lstm_mha_hpo(
        imf:int,
        data_dir:str,
        seq_len:int=24*7*4,
        input_size:int=8,
        num_splits:int=2,
        patience:int=5,
        max_epochs:int=50,
        reduction_factor:int=3,
        tpe_kwargs:dict=None,
        study_kwargs:dict=None,
        optimize_kwargs:dict=None
    ):

    tpe_kwargs = tpe_kwargs or {'seed':0}
    study_kwargs = study_kwargs or {}
    optimize_kwargs = optimize_kwargs or {'n_trials':100}

    def objective(trial:optuna.trial.Trial):

        #### general params ####
        batch_size = trial.suggest_int('batch_size', 16, 64, log=True)
        lr_init = trial.suggest_float('lr_init', 1e-4, 1e-1, log=True)

        #### TCN params ####
        dilation_base = 2
        kernel_size = trial.suggest_int('kernel_size', 2, 8)
        num_blocks = int(
            np.ceil(np.log2( ((seq_len-1)*(dilation_base-1)) / ((kernel_size-1)*2) + 1 ))
        )

        channel_sizes = list(np.zeros(num_blocks, dtype=int))
        for i in range(num_blocks):
            channel_sizes[i] = trial.suggest_int(
                f'channel_size_{num_blocks}_{i}',
                channel_sizes[i-1] if i>0 else 8,
                256
            )

        tcn_dropout = trial.suggest_float('tcn_dropout', 0.0, 0.5)

        #### LSTM params ####
        hidden_size0 = trial.suggest_int(
            'hidden_size0',
            max(16, channel_sizes[-1]//4),
            min(256, channel_sizes[-1])
        )
        hidden_size1 = trial.suggest_int('hidden_size1', 16, min(256, channel_sizes[-1]))
        hidden_size2 = trial.suggest_int('hidden_size2', 16, min(256, channel_sizes[-1]))
        hidden_sizes = [hidden_size0, hidden_size1, hidden_size2]

        lstm_dropout0 = trial.suggest_float('lstm_dropout0', 0.0, 0.5)
        lstm_dropout1 = trial.suggest_float('lstm_dropout1', 0.0, 0.5)
        lstm_dropouts = [lstm_dropout0, lstm_dropout1]

        #### MHA params ####
        mha_dropout = trial.suggest_float('mha_dropout', 0.0, 0.5)
        mha_heads = trial.suggest_categorical('mha_heads', [1, 2, 4, 8, 16])
        if hidden_sizes[-1]%mha_heads != 0:
            raise optuna.TrialPruned()

        #### callbacks ####
        callbacks = [
            optuna.integration.PyTorchLightningPruningCallback(trial, monitor='val_loss'),
            pl.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        ]

        #### training ####
        val_losses = np.zeros(num_splits)

        for i in range(1, num_splits+1):
            datamodule = EPFDataModule(
                X_train_path=f'{data_dir}df_train{i}_scaled.pkl',
                X_val_path=f'{data_dir}df_val{i}_scaled.pkl',
                y_train_path=f'{data_dir}df_train{i}_imf_scaled.pkl',
                y_val_path=f'{data_dir}df_val{i}_imf_scaled.pkl',
                imf=imf,
                batch_size=batch_size
            )
            mod = TCN_LSTM_MHA(
                input_size=input_size,
                channel_sizes=channel_sizes,
                kernel_size=kernel_size,
                hidden_sizes=hidden_sizes,
                tcn_dropout=tcn_dropout,
                lstm_dropouts=lstm_dropouts,
                mha_dropout=mha_dropout,
                mha_heads=mha_heads,
                lr_init=lr_init
            )
            
            trainer = pl.Trainer(
                max_epochs=50,
                callbacks=callbacks,
                accelerator='gpu',
                logger=False,
                enable_checkpointing=False
            )
            trainer.fit(mod, datamodule=datamodule)
            val_losses[i-1] = trainer.callback_metrics['val_loss'].item()

        return np.mean(val_losses)
    
    sampler = optuna.samplers.TPESampler(**tpe_kwargs)
    pruner = optuna.pruners.HyperbandPruner(min_resource=patience, max_resource=max_epochs, reduction_factor=reduction_factor)
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner, **study_kwargs)
    study.optimize(objective, **optimize_kwargs)

    return study