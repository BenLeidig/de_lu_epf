#!/bin/bash
#SBATCH --job-name=tune_en
#SBATCH --output=/u/$USER/de_lu_epf/logs/slurm/out/tune_en.out
#SBATCH --error=/u/$USER/de_lu_epf/logs/slurm/err/tune_en.err
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bleidig2@illinois.edu
#SBATCH --time=168:00:00
#SBATCH --array=0-23

USER_DIR=/u/$USER

source $USER_DIR/venvs/.venv/bin/activate
srun python $USER_DIR/de_lu_epf/src/hpo/tune_dmf/tune_en.py