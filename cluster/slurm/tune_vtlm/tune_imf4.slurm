#!/bin/bash
#SBATCH --job-name=vtlm_tune_imf4
#SBATCH --output=/%u/de_lu_epf/logs/slurm/out/vtlm_tune_imf4.out
#SBATCH --error=/%u/de_lu_epf/logs/slurm/err/vtlm_tune_imf4.err
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bleidig2@illinois.edu
#SBATCH --time=168:00:00

USER_DIR=/u/$USER

source USER_DIR/venvs/.venv/bin/activate
srun python USER_DIR/de_lu_epf/src/hpo/tune_vtlm/tune_imf4.py