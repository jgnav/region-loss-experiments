#!/bin/bash -l
#SBATCH --job-name=ibot_resume_5383_8gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=standard-gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:l40s:4

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

CONFIG_PATH="${1:-train_resume_1295383_8gpu.yaml}"
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Training configuration not found: ${CONFIG_PATH}" >&2
    exit 2
fi

module --force purge
module load apps/2021
module load Python/3.10.8-GCCcore-12.2.0

# shellcheck disable=SC1091
source .venv/bin/activate

export OMP_NUM_THREADS=1
export WANDB_MODE=online
export IBOT_RUN_ID="${IBOT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-slurm-${SLURM_JOB_ID}}"
export WANDB_API_KEY="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"
export CONFIG_PATH

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | sed -n '1p')"
MASTER_PORT="$((20000 + SLURM_JOB_ID % 20000))"
export MASTER_ADDR MASTER_PORT

echo "Config:       ${CONFIG_PATH}"
echo "Output run:   ${IBOT_RUN_ID}"
echo "W&B run ID:   ${WANDB_RUN_ID:-new run}"
echo "W&B resume:   ${WANDB_RESUME:-disabled}"
echo "Nodes:        ${SLURM_NNODES}"
echo "Master:       ${MASTER_ADDR}:${MASTER_PORT}"

srun --kill-on-bad-exit=1 bash -c '
    torchrun \
        --nnodes="${SLURM_NNODES}" \
        --nproc-per-node=4 \
        --node-rank="${SLURM_NODEID}" \
        --master-addr="${MASTER_ADDR}" \
        --master-port="${MASTER_PORT}" \
        train.py "${CONFIG_PATH}"
'
