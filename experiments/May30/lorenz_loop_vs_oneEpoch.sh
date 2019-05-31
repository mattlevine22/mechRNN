#!/bin/bash

#Submit this script with: sbatch train_submission

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=30G   # memory per CPU core
#SBATCH -c 1 # 1 core per task

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load cuda/9.0
module load python/2.7.15-tf

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_simple.py --savedir lorenz63_loop --epoch 100000 --delta_t 0.1 --t_end 100 --train_frac 0.9 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_simple.py --savedir lorenz63_oneEpoch --epoch 1 --delta_t 0.1 --t_end 10000000 --train_frac 0.999999 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_perturbTrivial.py --savedir lorenz63_loop_perturbTrivial --epoch 1000 --delta_t 0.1 --t_end 100 --train_frac 0.9 --n_perturbations 2 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_perturbTrivial.py --savedir lorenz63_oneEpoch_perturbTrivial --epoch 1 --delta_t 0.1 --t_end 100000 --train_frac 0.9999 --n_perturbations 2
wait