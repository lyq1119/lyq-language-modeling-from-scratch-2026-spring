#!/bin/bash
#SBATCH --job-name=a4token
#SBATCH --partition=lcpu-infra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=60000M
#SBATCH --time=01:00:00
#SBATCH --output=/home/lcpu/16443960/assignment4-data/logs/tokenize_%j.out
#SBATCH --error=/home/lcpu/16443960/assignment4-data/logs/tokenize_%j.err

set -euo pipefail
cd "$HOME/assignment4-data"
mkdir -p logs
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=1
echo "host=$(hostname) start=$(date)"
.venv/bin/python -m cc2026.tokenize_data --workers 48 "$@"
echo "done=$(date)"
