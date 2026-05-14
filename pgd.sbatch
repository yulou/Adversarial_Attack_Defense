#!/bin/bash
#SBATCH --account=mscaisuperpod
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00

conda activate arin5203_a3

python pgd_training.py