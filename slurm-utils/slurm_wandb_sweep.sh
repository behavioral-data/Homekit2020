#!/bin/bash
#SBATCH --job-name=wandb_sweep_$1_$2
#SBATCH --output=/mmfs1/gscratch/bdata/mikeam/MobileSensingSuite/slurm-utils/jobs/%x-%j.log
#SBATCH --error=/mmfs1/gscratch/bdata/mikeam/MobileSensingSuite/slurm-utils/jobs/%x-%j.out
#SBATCH --account=bdata
#SBATCH --partition=gpu-rtx6k
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --mem=45G
### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --chdir=/gscratch/bdata/mikeam/MobileSensingSuite
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate MobileSensingSuite
wandb agent $1 --count 1 
