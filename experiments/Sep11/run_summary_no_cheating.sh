#!/bin/bash

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load cuda/9.0
module load python/2.7.15-tf

python ../scripts/lorenz63_3dvar_summarize.py --output_dir 3DVAR_learning_no_cheat_V1 --n_train_trajectories 10 --n_test_trajectories 10
