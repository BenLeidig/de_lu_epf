#!/bin/bash
#SBATCH --job-name=tune_vl_imf%a
#SBATCH --output=/u/%u/de_lu_epf/logs/slurm/out/tune_vl_imf%a.out
#SBATCH --error=/u/%u/de_lu_epf/logs/slurm/err/tune_vl_imf%a.err
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
#SBATCH --array=0-5

USER_DIR=/u/$USER

source $USER_DIR/venvs/.venv/bin/activate
export PYTHONPATH=$USER_DIR/de_lu_epf
srun python $USER_DIR/de_lu_epf/scripts/models/hpo/hybrid/tune_vl.py