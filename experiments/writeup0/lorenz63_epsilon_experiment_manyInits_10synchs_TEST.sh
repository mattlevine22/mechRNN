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

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/prediction_with_model_error/Init0 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/prediction_with_model_error/Init1 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/prediction_with_model_error/Init2 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/prediction_with_model_error/Init3 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/prediction_with_model_error/Init4 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/prediction_with_model_error/Init5 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/prediction_with_model_error/Init6 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/prediction_with_model_error/Init7 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/prediction_with_model_error/Init8 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/prediction_with_model_error/Init9 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10
wait