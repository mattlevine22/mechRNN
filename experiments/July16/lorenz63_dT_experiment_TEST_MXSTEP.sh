#!/bin/bash

#Submit this script with: sbatch train_submission

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=3G   # memory per CPU core
#SBATCH -c 1 # 1 core per task

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load cuda/9.0
module load python/2.7.15-tf


srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --n_tests=10 --savedir=Test_dT_constant_LENGTH_defaultMXSTEP &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --n_tests=10 --savedir=Test_dT_constant_LENGTH_5000000MXSTEP --mxstep=5000000 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --n_tests=10 --n_train_points=1000 --savedir=Test_dT_constant_NTRAIN_defaultMXSTEP &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --n_tests=10 --n_train_points=1000 --savedir=Test_dT_constant_NTRAIN_5000000MXSTEP --mxstep=5000000
wait