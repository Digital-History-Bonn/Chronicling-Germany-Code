#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=A40devel
#SBATCH --job-name=convert_newspaper
#SBATCH --error=slurm/job.%J.err
#SBATCH --output=slurm/job.%J.out
#SBATCH --mail-user=s6cnscul@uni-bonn.de
#SBATCH --mail-type=ALL

ml Python 
source virtualpython/pytorch/bin/activate
cd /home/s6cnscul/repositories/NewspaperImageSegmentation/
PYTHONPATH=.
python script/convert_xml.py 