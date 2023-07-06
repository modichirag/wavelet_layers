#!/bin/bash -l

#SBATCH -p gpu
#SBATCH -t 12:00:00
#SBATCH -C a100-40gb
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH -J wq64L6v2

module --force purge
module load modules/2.1.1-20230405
module load cuda cudnn gcc

source activate jaxenv

cd ..

time python -u wavelet_sadapt.py 
