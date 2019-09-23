#!/bin/bash

#Submit this script with: sbatch train_submission

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=3G   # memory per CPU core
#SBATCH -c 1 # 1 core per task

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load cuda/9.0
module load python/2.7.15-tf

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST/Init0 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST/Init1 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST/Init2 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST/Init3 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST/Init4 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST/Init5 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST/Init6 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST/Init7 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST/Init8 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST/Init9 --noisy_training True --noise_frac 0.01 --epoch 1000 --delta_t 0.1 --t_train 100 --n_tests 10
wait