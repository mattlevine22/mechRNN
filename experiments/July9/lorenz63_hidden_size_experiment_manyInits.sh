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

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes/Init0 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.333 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes/Init1 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.333 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes/Init2 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.333 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes/Init3 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.333 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes/Init4 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.333 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes/Init5 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.333 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes/Init6 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.333 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes/Init7 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.333 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes/Init8 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.333 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_hiddensizes.py --savedir lorenz63_hiddensizes/Init9 --epoch 10000 --delta_t 0.1 --t_end 300 --train_frac 0.333 --random_state_inits True --compute_kl False
wait