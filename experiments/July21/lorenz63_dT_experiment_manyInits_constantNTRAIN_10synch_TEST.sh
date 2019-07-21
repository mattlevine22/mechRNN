#!/bin/bash

#Submit this script with: sbatch train_submission

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=10G   # memory per CPU core
#SBATCH -c 1 # 1 core per task

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load cuda/9.0
module load python/2.7.15-tf


# srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --n_tests=3 --t_train=100 --savedir=Test_dT_constant_LENGTH_V2 &
# srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --n_tests=3 --n_train_points=1000 --savedir=Test_dT_constant_NTRAIN_V2


# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_constantNUMTRAIN_10synchs_TEST/Init0 --epoch 1000 --n_train_points 1000 --n_tests=10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_constantNUMTRAIN_10synchs_TEST/Init1 --epoch 1000 --n_train_points 1000 --n_tests=10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_constantNUMTRAIN_10synchs_TEST/Init2 --epoch 1000 --n_train_points 1000 --n_tests=10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_constantNUMTRAIN_10synchs_TEST/Init3 --epoch 1000 --n_train_points 1000 --n_tests=10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_constantNUMTRAIN_10synchs_TEST/Init4 --epoch 1000 --n_train_points 1000 --n_tests=10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_constantNUMTRAIN_10synchs_TEST/Init5 --epoch 1000 --n_train_points 1000 --n_tests=10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_constantNUMTRAIN_10synchs_TEST/Init6 --epoch 1000 --n_train_points 1000 --n_tests=10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_constantNUMTRAIN_10synchs_TEST/Init7 --epoch 1000 --n_train_points 1000 --n_tests=10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_constantNUMTRAIN_10synchs_TEST/Init8 --epoch 1000 --n_train_points 1000 --n_tests=10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_constantNUMTRAIN_10synchs_TEST/Init9 --epoch 1000 --n_train_points 1000 --n_tests=10
wait