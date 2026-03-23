#!/bin/bash

# Target columns to create TCN-LSTM-MHA networks for
target_col=("imf1" "imf2" "imf3" "imf4" "imf5" "imf_resid")

# Create a TCN-LSTM-MHA network for each column
for imf in "${target_col[@]}"; do
    sbatch "cluster/slurm/tune_{imf}.slurm"
done