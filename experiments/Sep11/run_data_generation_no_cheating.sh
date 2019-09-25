#!/bin/bash

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load cuda/9.0
module load python/2.7.15-tf

python ../scripts/lorenz63_3dvar_generateData.py --n_trajectories 10 --output_dir /groups/astuart/mlevine/Sep11/3DVAR_learning_no_cheat_V2_50 --output_filename TrainData.npz --t_end 2000
python ../scripts/lorenz63_3dvar_generateData.py --n_trajectories 10 --output_dir /groups/astuart/mlevine/Sep11/3DVAR_learning_no_cheat_V2_50 --output_filename TestData.npz --t_end 5
