#!/bin/bash

#Submit this script with: sbatch train_submission

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=30G   # memory per CPU core
#SBATCH -c 1 # 1 core per task

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load cuda/9.0
module load python/2.7.15-tf

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_inv_meas/Init0 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.33 --n_experiments 10 --random_state_inits True --compute_kl True &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_inv_meas/Init1 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.33 --n_experiments 10 --random_state_inits True --compute_kl True &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_inv_meas/Init2 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.33 --n_experiments 10 --random_state_inits True --compute_kl True &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_inv_meas/Init3 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.33 --n_experiments 10 --random_state_inits True --compute_kl True &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_inv_meas/Init4 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.33 --n_experiments 10 --random_state_inits True --compute_kl True &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_inv_meas/Init5 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.33 --n_experiments 10 --random_state_inits True --compute_kl True &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_inv_meas/Init6 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.33 --n_experiments 10 --random_state_inits True --compute_kl True &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_inv_meas/Init7 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.33 --n_experiments 10 --random_state_inits True --compute_kl True &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_inv_meas/Init8 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.33 --n_experiments 10 --random_state_inits True --compute_kl True &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps.py --savedir lorenz63_eps_inv_meas/Init9 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.33 --n_experiments 10 --random_state_inits True --compute_kl True
wait