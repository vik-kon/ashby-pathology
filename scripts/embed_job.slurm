#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --account=bfqi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

source ~/ashby-pathology/.venv/bin/activate
cd ~/ashby-pathology
export HF_HOME=/work/nvme/bfqi/$USER/hf
export HF_HUB_OFFLINE=1
python -m src.embed