#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=yolo
#SBATCH --mem=20000

module purge

module load Python/3.9.6-GCCcore-11.2.0

source ~/env/bin/activate

python tests.py
