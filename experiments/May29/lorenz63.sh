#!/bin/bash

#Submit this script with: sbatch train_submission

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=20G   # memory per CPU core
#SBATCH -c 1 # 1 core per task

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load cuda/9.0
module load python/2.7.15-tf

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63/loop_10000epochs_eps_iter0 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63/loop_10000epochs_eps_iter1 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63/loop_10000epochs_eps_iter2 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63/loop_10000epochs_eps_iter3 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63/loop_10000epochs_eps_iter4 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63/loop_10000epochs_eps_iter5 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63/loop_10000epochs_eps_iter6 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63/loop_10000epochs_eps_iter7 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63/loop_10000epochs_eps_iter8 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63/loop_10000epochs_eps_iter9 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9
wait