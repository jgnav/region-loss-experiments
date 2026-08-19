#!/bin/bash -l
#SBATCH --job-name=ibot_evaluation
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=standard-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:l40s:4

set -euo pipefail

if (( $# > 2 )); then
    echo "Usage: sbatch slurm_evaluation.sh [DATASETS_ROOT] [OUTPUT_DIR]" >&2
    exit 2
fi

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${REPO_DIR}"

# Edit this line to select the checkpoint to evaluate.
# CHECKPOINT_PATH="${REPO_DIR}/output/region_loss/20260818T233720Z-slurm-1295841/checkpoint.pth"
CHECKPOINT_PATH="${REPO_DIR}/checkpoints/ibot_vit_small.pth"
DATASETS_ROOT="${1:-${REPO_DIR}/dataset}"
OUTPUT_DIR="${2:-${REPO_DIR}/output/evaluation/slurm-${SLURM_JOB_ID}}"
VENV_DIR="${EVALUATION_VENV:-${REPO_DIR}/.venv}"

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
    echo "Create it with: sbatch slurm_setup_env.sh" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"

module --force purge
module load apps/2021
module load Python/3.10.8-GCCcore-12.2.0

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# RAPIDS/cuML loads CUDA libraries dynamically. Make the CUDA 12 libraries
# installed with the virtual environment visible before Python starts.
NVIDIA_LIB_ROOT="${VENV_DIR}/lib/python3.10/site-packages/nvidia"
for package in cublas cuda_nvrtc cuda_runtime cufft curand cusolver cusparse nvjitlink; do
    library_dir="${NVIDIA_LIB_ROOT}/${package}/lib"
    if [[ -d "${library_dir}" ]]; then
        export LD_LIBRARY_PATH="${library_dir}:${LD_LIBRARY_PATH:-}"
    fi
done

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
echo "Python:          $(command -v python)"
nvidia-smi --list-gpus

srun --ntasks=1 --cpu-bind=cores \
    python -u evaluation/full-evaluation \
    "${CHECKPOINT_PATH}" \
    --datasets-root "${DATASETS_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-workers 8
