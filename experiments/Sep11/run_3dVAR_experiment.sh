#!/bin/bash

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load cuda/9.0
module load python/2.7.15-tf

python ../scripts/lorenz63_3dvar_generateData.py --n_trajectories 10 --output_dir 3DVAR_learning_V1 --output_filename TrainData.npz --t_end 500
python ../scripts/lorenz63_3dvar_generateData.py --n_trajectories 10 --output_dir 3DVAR_learning_V1 --output_filename TestData.npz --t_end 5

sbatch jobs_3DVAR_experiments.sh

python ../scripts/lorenz63_3dvar_summarize.py --n_train_trajectories 10 --n_test_trajectories 10
