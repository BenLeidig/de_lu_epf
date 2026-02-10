#!/bin/bash
#SBATCH --job-name=vmd_corr_solver
#SBATCH --output=/scratch/%u/vmd_corr_solver.out
#SBATCH --error=/scratch/%u/vmd_corr_solver.err
#SBATCH --account=stat
#SBATCH --partition=stat
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bleidig2@illinois.edu

source /u/bleidig2/venvs/.venv/bin/activate
srun python /u/bleidig2/vmd_corr_solver.py