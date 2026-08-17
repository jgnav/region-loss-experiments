#!/bin/bash -l
#SBATCH --job-name=ibot_debug
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=debug-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:a100:4

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

module --force purge
module load apps/2021
module load Python/3.10.8-GCCcore-12.2.0

# shellcheck disable=SC1091
source .venv/bin/activate

export OMP_NUM_THREADS=1
export WANDB_MODE=online
: "${WANDB_API_KEY:?Export WANDB_API_KEY before submitting the job}"

srun torchrun --standalone --nproc-per-node=4 train.py train.yaml
