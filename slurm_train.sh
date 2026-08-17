#!/bin/bash -l
#SBATCH --job-name=ibot_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=standard-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=1G
#SBATCH --time=03:00:00
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
export WANDB_API_KEY="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"

srun torchrun --standalone --nproc-per-node=4 train.py train.yaml
