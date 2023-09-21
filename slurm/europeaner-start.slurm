#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres gpu:4
#SBATCH --ntasks-per-node=64
#SBATCH --partition=A100medium
#SBATCH --job-name=dh_segment_newspaper
#SBATCH --error=slurm/job.%J.err
#SBATCH --output=slurm/job.%J.out
#SBATCH --mail-user=s6cnscul@uni-bonn.de
#SBATCH --mail-type=ALL

cd /home/s6cnscul/repositories/NewspaperImageSegmentation/
eval "$(conda shell.bash hook)"
conda activate c_torch
python src/news_seg/train.py -e 1 -n dh_segment_A100_test -b 64 -d ../../data/newspaper/  -l BritTrainB_best -w 64