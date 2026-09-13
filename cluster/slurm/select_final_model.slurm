#!/bin/bash
#SBATCH --job-name=select_final_model
#SBATCH --output=/u/%u/de_lu_epf/logs/slurm/out/select_final_model.out
#SBATCH --error=/u/%u/de_lu_epf/logs/slurm/err/select_final_model.err
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bleidig2@illinois.edu
#SBATCH --time=24:00:00

# Requests a GPU since the winning candidate may be an ANN/hybrid
# architecture, in which case this job performs its train_val retrain here
# (never locally - see scripts/models/select_final_model.py's docstring).
# If the winner turns out to be a DMF model instead, the GPU simply goes
# unused for that retrain step, which is cheap classical-ML fitting.
#
# Prerequisite: every candidate's val predictions must already exist (i.e.
# HPO has completed for ann/dmf/hybrid and all 18 predict jobs have run)
# before submitting this job.

USER_DIR=/u/$USER

source $USER_DIR/venvs/.venv/bin/activate
export PYTHONPATH=$USER_DIR/de_lu_epf
srun python $USER_DIR/de_lu_epf/scripts/models/select_final_model.py
