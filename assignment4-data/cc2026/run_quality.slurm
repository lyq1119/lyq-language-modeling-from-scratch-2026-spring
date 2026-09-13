#!/bin/bash
#SBATCH --job-name=a4quality
#SBATCH --partition=lcpu-infra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32000M
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/16443960/assignment4-data/logs/quality_%j.out
#SBATCH --error=/home/lcpu/16443960/assignment4-data/logs/quality_%j.err

set -euo pipefail
cd "$HOME/assignment4-data"
mkdir -p logs
export OMP_NUM_THREADS=1
echo "host=$(hostname) start=$(date)"
.venv/bin/python -m cc2026.train_quality_classifier "$@"
echo "done=$(date)"
