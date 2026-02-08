#!/bin/bash
#SBATCH -p ai
#SBATCH -J build_model
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2023.09
source activate py312
module load mpi/openmpi/5.0.0-gcc-11.4.0-cuda12.2
export LD_LIBRARY_PATH=/HOME/paratera_xy/pxy550/.conda/envs/py312/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/HOME/paratera_xy/pxy550/.conda/envs/py312/bin/python:$PYTHONPATH
export PYTHONUNBUFFERED=1

date
python -u evaluation_train.py > evaluation_train.out
python -u evaluation_test.py > evaluation_test.out
date
