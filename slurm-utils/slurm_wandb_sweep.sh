#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH --output=slurm_out/%x-%j.log
#SBATCH --error=slurm_out/%x-%j.out
#SBATCH --account=bdata
#SBATCH --partition=gpu-rtx6k
#SBATCH --time=24:00:00
### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=1
#SBATCH --mem=45G
### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --chdir=/gscratch/bdata/mikeam/SeattleFluStudy

conda activate seattleflustudy
wandb agent $1 

