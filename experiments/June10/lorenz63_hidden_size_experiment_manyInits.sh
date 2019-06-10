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
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes_loop_10000epochs_v2/Init0 --epoch 10000 --delta_t 0.1 --t_end 120 --train_frac 0.75 --random_state_inits True &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes_loop_10000epochs_v2/Init1 --epoch 10000 --delta_t 0.1 --t_end 120 --train_frac 0.75 --random_state_inits True &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes_loop_10000epochs_v2/Init2 --epoch 10000 --delta_t 0.1 --t_end 120 --train_frac 0.75 --random_state_inits True &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes_loop_10000epochs_v2/Init3 --epoch 10000 --delta_t 0.1 --t_end 120 --train_frac 0.75 --random_state_inits True &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes_loop_10000epochs_v2/Init4 --epoch 10000 --delta_t 0.1 --t_end 120 --train_frac 0.75 --random_state_inits True &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes_loop_10000epochs_v2/Init5 --epoch 10000 --delta_t 0.1 --t_end 120 --train_frac 0.75 --random_state_inits True &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes_loop_10000epochs_v2/Init6 --epoch 10000 --delta_t 0.1 --t_end 120 --train_frac 0.75 --random_state_inits True &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes_loop_10000epochs_v2/Init7 --epoch 10000 --delta_t 0.1 --t_end 120 --train_frac 0.75 --random_state_inits True &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes_loop_10000epochs_v2/Init8 --epoch 10000 --delta_t 0.1 --t_end 120 --train_frac 0.75 --random_state_inits True &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes_loop_10000epochs_v2/Init9 --epoch 10000 --delta_t 0.1 --t_end 120 --train_frac 0.75 --random_state_inits True
wait