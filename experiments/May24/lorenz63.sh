#!/bin/bash

#Submit this script with: sbatch train_submission

#SBATCH --time=36:00:00   # walltime
#SBATCH --ntasks=2   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=25G   # memory per CPU core
#SBATCH -c 1 # 1 core per task

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load cuda/9.0
module load python/2.7.15-tf

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63/loop_10000epochs_eps --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63/loop_10000epochs_hs --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 &
wait