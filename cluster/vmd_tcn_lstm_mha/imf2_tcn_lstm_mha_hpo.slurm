#!/bin/bash
#SBATCH --job-name=imf2_hpo
#SBATCH --output=/scratch/%u/imf2_tcn_lstm_mha_hpo.out
#SBATCH --error=/scratch/%u/imf2_tcn_lstm_mha_hpo.err
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

source /u/bleidig2/venvs/.venv/bin/activate
srun python /u/bleidig2/scratch/imf2_tcn_lstm_mha_hpo.py