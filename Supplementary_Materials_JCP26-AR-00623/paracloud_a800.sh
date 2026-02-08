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
python -u mace_convert_virial_to_stress.py

mace_run_train \
    --name="MACE_MPA_ft" \
    --foundation_model="./mace-mpa-0-medium.model" \
    --multiheads_finetuning=False \
    --valid_fraction=0.05 \
    --train_file="train_mace.extxyz" \
    --test_file="test_mace.extxyz" \
    --energy_key="REF_energy" \
    --forces_key="REF_forces" \
    --stress_key="REF_stress" \
    --compute_stress=True \
    --loss="universal" \
    --energy_weight=1.0 \
    --forces_weight=10.0 \
    --stress_weight=1.0 \
    --E0s="average" \
    --lr=0.001 \
    --scaling="rms_forces_scaling" \
    --batch_size=12 \
    --max_num_epochs=1000 \
    --ema \
    --ema_decay=0.999 \
    --amsgrad \
    --default_dtype="float64" \
    --device=cuda \
    --seed=3 \
    --enable_cueq=True \
    --error_table="PerAtomRMSEstressvirials"
date
