#!/usr/bin/env bash
set -euo pipefail

# Usage: ./profile_nsys.sh [model-size] [context-length] [mode] [CUDA device index]
model_size="${1:-small}"
context_length="${2:-512}"
mode="${3:-full}"
cuda_device="${4:-0}"

project_nsys="$(dirname "$0")/.venv/bin/nsys"
if [[ -x "${project_nsys}" ]]; then
    nsys_bin="${project_nsys}"
else
    nsys_bin="$(command -v nsys || true)"
fi
if [[ -z "${nsys_bin}" ]]; then
    for candidate in /usr/local/cuda-12.6/bin/nsys /usr/local/cuda/bin/nsys /usr/local/cuda-11.8/bin/nsys; do
        if [[ -x "${candidate}" ]]; then
            nsys_bin="${candidate}"
            break
        fi
    done
fi
if [[ -z "${nsys_bin}" ]]; then
    echo "Could not find nsys. Install NVIDIA Nsight Systems or add it to PATH." >&2
    exit 1
fi

mkdir -p profiles
report="profiles/${model_size}_ctx${context_length}_${mode}"

CUDA_VISIBLE_DEVICES="${cuda_device}" "${nsys_bin}" profile \
    --trace=cuda,nvtx,osrt,cublas \
    --sample=none \
    --force-overwrite=true \
    --output="${report}" \
    uv run python benchmark.py \
        --device cuda \
        --model-size "${model_size}" \
        --context-length "${context_length}" \
        --mode "${mode}" \
        --warmup-steps 2 \
        --measurement-steps 1 \
        --json

"${nsys_bin}" stats \
    --force-export=true \
    --filter-nvtx=benchmark \
    --report=cuda_gpu_kern_sum \
    "${report}.nsys-rep" > "${report}_stats.txt"

echo "Report: ${report}.nsys-rep"
echo "Summary: ${report}_stats.txt"
