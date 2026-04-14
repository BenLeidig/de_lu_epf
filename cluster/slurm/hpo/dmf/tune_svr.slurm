#!/bin/bash
#SBATCH --job-name=tune_svr_hr%a
#SBATCH --output=/u/%u/de_lu_epf/logs/slurm/out/tune_svr_hr%a.out
#SBATCH --error=/u/%u/de_lu_epf/logs/slurm/err/tune_svr_hr%a.err
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bleidig2@illinois.edu
#SBATCH --time=48:00:00
#SBATCH --array=0-23

USER_DIR=/u/$USER

source $USER_DIR/venvs/.venv/bin/activate
export PYTHONPATH=$USER_DIR/de_lu_epf
srun python $USER_DIR/de_lu_epf/scripts/models/hpo/dmf/tune_svr.py