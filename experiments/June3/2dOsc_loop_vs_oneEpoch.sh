#!/bin/bash

#Submit this script with: sbatch train_submission

#SBATCH --time=6:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=10G   # memory per CPU core
#SBATCH -c 1 # 1 core per task

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load cuda/9.0
module load python/2.7.15-tf

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python ../scripts/run_script_2dOscillator_SCRIPTSTYLE_simple.py --savedir 2dOscillator_run1/loop_0.8 --epoch 200 --delta_t 0.1 --t_end 100 --train_frac 0.8
# srun --exclusive -N 1 -n 1 python ../scripts/run_script_2dOscillator_SCRIPTSTYLE_simple.py --savedir 2dOscillator_run1/oneEpoch --epoch 1 --delta_t 0.1 --t_end 2000 --train_frac 0.99 &
# srun --exclusive -N 1 -n 1 python ../scripts/run_script_2dOscillator_SCRIPTSTYLE_perturbTrivial.py --savedir 2dOscillator_run1/loop_perturbTrivial --epoch 200 --delta_t 0.1 --t_end 100 --train_frac 0.9 --n_perturbations 2 &
# srun --exclusive -N 1 -n 1 python ../scripts/run_script_2dOscillator_SCRIPTSTYLE_perturbTrivial.py --savedir 2dOscillator_run1/oneEpoch_perturbTrivial --epoch 1 --delta_t 0.1 --t_end 2000 --train_frac 0.99 --n_perturbations 2
wait