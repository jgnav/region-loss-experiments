#!/bin/bash -l
#SBATCH --job-name=ibot_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=standard-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:4

set -euo pipefail

if (( $# < 1 || $# > 3 )); then
    echo "Usage: sbatch slurm_evaluation.sh CHECKPOINT [DATASETS_ROOT] [OUTPUT_DIR]" >&2
    exit 2
fi

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${REPO_DIR}"

CHECKPOINT_PATH="${1:-${REPO_DIR}/output/region_loss/20260818T154524Z-slurm-1295383/checkpoint.pth}"
DATASETS_ROOT="${2:-${REPO_DIR}/datasets}"
OUTPUT_DIR="${3:-${REPO_DIR}/output/evaluation/slurm-${SLURM_JOB_ID}}"
VENV_DIR="${EVALUATION_VENV:-${REPO_DIR}/.venv-evaluation}"

if [[ "${CHECKPOINT_PATH}" != /* ]]; then
    CHECKPOINT_PATH="${REPO_DIR}/${CHECKPOINT_PATH}"
fi
if [[ "${DATASETS_ROOT}" != /* ]]; then
    DATASETS_ROOT="${REPO_DIR}/${DATASETS_ROOT}"
fi
if [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${REPO_DIR}/${OUTPUT_DIR}"
fi

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
    exit 2
fi
if [[ ! -d "${DATASETS_ROOT}" ]]; then
    echo "Datasets directory not found: ${DATASETS_ROOT}" >&2
    exit 2
fi
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "Evaluation environment not found: ${VENV_DIR}" >&2
    echo "Create it using the commands in evaluation/README.md before submitting." >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "Job:             ${SLURM_JOB_ID}"
echo "Node:            ${SLURMD_NODENAME:-unknown}"
echo "Visible GPUs:    ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Checkpoint:      ${CHECKPOINT_PATH}"
echo "Datasets:        ${DATASETS_ROOT}"
echo "Output:          ${OUTPUT_DIR}"
echo "Python:          ${VENV_DIR}/bin/python"
nvidia-smi --list-gpus

srun --ntasks=1 --cpu-bind=cores \
    "${VENV_DIR}/bin/python" -u evaluation/full-evaluation \
    "${CHECKPOINT_PATH}" \
    --datasets-root "${DATASETS_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-workers 8
