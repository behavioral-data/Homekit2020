#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH --output=slurm_out/%x-%j.log
#SBATCH --error=slurm_out/%x-%j.out
#SBATCH --account=bdata
#SBATCH --partition=ckpt
#SBATCH --time=12:00:00
### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=1
#SBATCH --mem=90G
### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --chdir=/gscratch/bdata/mikeam/SeattleFluStudy
#SBATCH --gres=gpu:titan:2

export PATH=$PATH:/gscratch/bdata/mikeam/anaconda3/bin
source ~/.bashrc
conda activate seattleflustudy
wandb agent $1 

