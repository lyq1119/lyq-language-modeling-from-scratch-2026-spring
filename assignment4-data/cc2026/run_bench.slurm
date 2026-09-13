#!/bin/bash
#SBATCH --job-name=a4bench
#SBATCH --partition=lcpu-infra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=60000M
#SBATCH --time=00:20:00
#SBATCH --output=/home/lcpu/16443960/assignment4-data/logs/bench_%j.out
#SBATCH --error=/home/lcpu/16443960/assignment4-data/logs/bench_%j.err

set -euo pipefail
cd "$HOME/assignment4-data"
mkdir -p logs
export HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1   # 该集群 RoCE IB 插件会让 NCCL 段错误；单节点用 SHM 即可

TRAIN_BIN="${TRAIN_BIN:-$HOME/assignment4-data/local-shared-data/tokenized/train.bin}"
VALID_BIN="${VALID_BIN:-$HOME/assignment4-data/local-shared-data/tokenized_paloma_c4_100_domains_validation.bin}"
MODEL_OUT="${MODEL_OUT:-$HOME/assignment4-data/local-shared-data/runs/bench}"
STEPS="${STEPS:-40}"
EVAL_INTERVAL="${EVAL_INTERVAL:-20}"
EVAL_ITERS="${EVAL_ITERS:-5}"
MICRO_BATCH="${MICRO_BATCH:-64}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EVAL_BATCH="${EVAL_BATCH:-32}"

echo "host=$(hostname) start=$(date) steps=$STEPS"
nvidia-smi -L
NO_COMPILE="${NO_COMPILE:-1}"
EXTRA=""
if [ "$NO_COMPILE" = "1" ]; then EXTRA="--no-compile"; fi
time .venv/bin/torchrun --standalone --nproc_per_node=8 -m cc2026.train_local \
    --train-bin "$TRAIN_BIN" \
    --valid-bin "$VALID_BIN" \
    --model-output "$MODEL_OUT" \
    --train-steps "$STEPS" \
    --eval-interval "$EVAL_INTERVAL" \
    --eval-iterations "$EVAL_ITERS" \
    --micro-batch-size "$MICRO_BATCH" \
    --grad-accum "$GRAD_ACCUM" \
    --eval-batch-size "$EVAL_BATCH" \
    $EXTRA
echo "done=$(date)"
