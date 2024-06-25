#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J rays
#SBATCH -o rays.%J.out
#SBATCH -e rays.%J.err
#SBATCH --mail-user=yahia.battach@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=23:30:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --constraint=a100


#run the application:
module load cuda
conda activate deeplenv
srun python train.py --gpus 4 --workers 10


