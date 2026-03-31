#!/bin/bash

# Model classes to tune
model_class=("en" "lgbmr" "rfr" "svr" "xgbr")

# Submit job for each model class
for mod in "${model_class[@]}"; do
    sbatch "cluster/slurm/train/dmf/train_${mod}.slurm"
done