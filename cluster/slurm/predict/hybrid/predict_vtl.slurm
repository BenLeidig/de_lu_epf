#!/bin/bash
#SBATCH --job-name=predict_vtl
#SBATCH --output=/u/%u/de_lu_epf/logs/slurm/out/predict_vtl.out
#SBATCH --error=/u/%u/de_lu_epf/logs/slurm/err/predict_vtl.err
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bleidig2@illinois.edu
#SBATCH --time=2:00:00

USER_DIR=/u/$USER

source $USER_DIR/venvs/.venv/bin/activate
export PYTHONPATH=$USER_DIR/de_lu_epf
srun python $USER_DIR/de_lu_epf/scripts/models/predict/hybrid/predict_vtl.py