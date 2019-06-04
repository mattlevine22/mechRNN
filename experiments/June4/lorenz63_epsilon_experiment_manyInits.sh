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
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_loop_10000epochs/Init0 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 --n_experiments 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_loop_10000epochs/Init1 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 --n_experiments 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_loop_10000epochs/Init2 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 --n_experiments 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_loop_10000epochs/Init3 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 --n_experiments 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_loop_10000epochs/Init4 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 --n_experiments 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_loop_10000epochs/Init5 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 --n_experiments 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_loop_10000epochs/Init6 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 --n_experiments 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_loop_10000epochs/Init7 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 --n_experiments 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_loop_10000epochs/Init8 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 --n_experiments 10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_loop_10000epochs/Init9 --epoch 10000 --delta_t 0.1 --t_end 100 --train_frac 0.9 --n_experiments 10
wait