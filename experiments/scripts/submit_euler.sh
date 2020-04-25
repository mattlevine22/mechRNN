#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH -c 1 # 1 core per task

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load python3/3.7.0

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.00001/euler_test --delta_t 0.00001  --n_test_traj 1 --testcontinuous_ode_int_method Euler --psi0_ode_int_method Euler --datagen_ode_int_method Euler --K 9 --J 8 --F 10 --t_synch 5 --t_train 10 --t_invariant_measure 10 --n_subsample_gp 800 &
wait
