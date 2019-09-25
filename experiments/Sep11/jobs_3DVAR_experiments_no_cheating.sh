#!/bin/bash

#Submit this script with: sbatch train_submission

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=1G   # memory per CPU core
#SBATCH -c 1 # 1 core per task

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load cuda/9.0
module load python/2.7.15-tf

srun --exclusive -N 1 -n 1 python ../scripts/lorenz63_3dvar_train.py --output_dir 3DVAR_learning_no_cheat_V1 --cheat False --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 0 &
srun --exclusive -N 1 -n 1 python ../scripts/lorenz63_3dvar_train.py --output_dir 3DVAR_learning_no_cheat_V1 --cheat False --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 1 &
srun --exclusive -N 1 -n 1 python ../scripts/lorenz63_3dvar_train.py --output_dir 3DVAR_learning_no_cheat_V1 --cheat False --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 2 &
srun --exclusive -N 1 -n 1 python ../scripts/lorenz63_3dvar_train.py --output_dir 3DVAR_learning_no_cheat_V1 --cheat False --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 3 &
srun --exclusive -N 1 -n 1 python ../scripts/lorenz63_3dvar_train.py --output_dir 3DVAR_learning_no_cheat_V1 --cheat False --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 4 &
srun --exclusive -N 1 -n 1 python ../scripts/lorenz63_3dvar_train.py --output_dir 3DVAR_learning_no_cheat_V1 --cheat False --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 5 &
srun --exclusive -N 1 -n 1 python ../scripts/lorenz63_3dvar_train.py --output_dir 3DVAR_learning_no_cheat_V1 --cheat False --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 6 &
srun --exclusive -N 1 -n 1 python ../scripts/lorenz63_3dvar_train.py --output_dir 3DVAR_learning_no_cheat_V1 --cheat False --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 7 &
srun --exclusive -N 1 -n 1 python ../scripts/lorenz63_3dvar_train.py --output_dir 3DVAR_learning_no_cheat_V1 --cheat False --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 8 &
srun --exclusive -N 1 -n 1 python ../scripts/lorenz63_3dvar_train.py --output_dir 3DVAR_learning_no_cheat_V1 --cheat False --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 9
wait