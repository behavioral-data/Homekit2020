#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH --output=/mmfs1/gscratch/bdata/mikeam/MobileSensingSuite/slurm-utils/jobs/%x-%j.log
#SBATCH --error=/mmfs1/gscratch/bdata/mikeam/MobileSensingSuite/slurm-utils/jobs/%x-%j.out
#SBATCH --account=bdata
#SBATCH --partition=gpu-rtx6k
#SBATCH --time=2:00:00
### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=1
#SBATCH --mem=45G
### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --chdir=/gscratch/bdata/mikeam/MobileSensingSuite
#SBATCH --gres=gpu:1

#export PATH=$PATH:/gscratch/bdata/mikeam/anaconda3/bin
source ~/.bashrc
echo "submiting"
conda activate MobileSensingSuite
wandb agent $1 

