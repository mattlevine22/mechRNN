#!/bin/bash

#Submit this script with: sbatch train_submission

#SBATCH --time=6:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=10G   # memory per CPU core
#SBATCH -c 1 # 1 core per task

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load cuda/9.0
module load python/2.7.15-tf

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE.py --savedir lorenz63/one_epoch_v1 --epoch 1 --delta_t 0.01 --t_end 10000 --train_frac 0.9995 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE.py --savedir lorenz63/one_epoch_v2 --epoch 1 --delta_t 0.1 --t_end 10000 --train_frac 0.9995 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE.py --savedir lorenz63/one_epoch_v3 --epoch 1 --delta_t 0.1 --t_end 100000 --train_frac 0.99995 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE.py --savedir lorenz63/loop_epoch_v1 --epoch 100 --delta_t 0.01 --t_end 100 --train_frac 0.9
wait