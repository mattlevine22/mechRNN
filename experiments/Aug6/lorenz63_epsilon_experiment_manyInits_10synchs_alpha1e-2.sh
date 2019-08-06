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
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST_alpha1e-2/Init0 --alpha 0.01 --noisy_training True --noise_frac 0.01 --epoch 1 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST_alpha1e-2/Init1 --alpha 0.01 --noisy_training True --noise_frac 0.01 --epoch 1 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST_alpha1e-2/Init2 --alpha 0.01 --noisy_training True --noise_frac 0.01 --epoch 1 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST_alpha1e-2/Init3 --alpha 0.01 --noisy_training True --noise_frac 0.01 --epoch 1 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST_alpha1e-2/Init4 --alpha 0.01 --noisy_training True --noise_frac 0.01 --epoch 1 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST_alpha1e-2/Init5 --alpha 0.01 --noisy_training True --noise_frac 0.01 --epoch 1 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST_alpha1e-2/Init6 --alpha 0.01 --noisy_training True --noise_frac 0.01 --epoch 1 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST_alpha1e-2/Init7 --alpha 0.01 --noisy_training True --noise_frac 0.01 --epoch 1 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST_alpha1e-2/Init8 --alpha 0.01 --noisy_training True --noise_frac 0.01 --epoch 1 --delta_t 0.1 --t_train 100 --n_tests 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_withGP_10synchs_TEST_alpha1e-2/Init9 --alpha 0.01 --noisy_training True --noise_frac 0.01 --epoch 1 --delta_t 0.1 --t_train 100 --n_tests 10
wait