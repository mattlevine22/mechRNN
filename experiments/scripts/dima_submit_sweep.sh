#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH -c 1 # 1 core per task

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load python3/3.7.0

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.001/eps0.008_hx-2.0/tSynch50_tInvMeas100_nGpSub800 --hx -2.0 --eps 0.008 --delta_t 0.001 --K 9 --J 8 --F 10 --t_synch 50 --t_train 10 --t_invariant_measure 100 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.001/eps0.008_hx-2.0/tSynch500_tInvMeas2000_nGpSub800 --hx -2.0 --eps 0.008 --delta_t 0.001 --K 9 --J 8 --F 10 --t_synch 500 --t_train 10 --t_invariant_measure 2000 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.001/eps1_hx-2.0/tSynch50_tInvMeas100_nGpSub800 --hx -2.0 --eps 1 --delta_t 0.001 --K 9 --J 8 --F 10 --t_synch 50 --t_train 10 --t_invariant_measure 100 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.001/eps1_hx-2.0/tSynch500_tInvMeas2000_nGpSub800 --hx -2.0 --eps 1 --delta_t 0.001 --K 9 --J 8 --F 10 --t_synch 500 --t_train 10 --t_invariant_measure 2000 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.001/eps10_hx-2.0/tSynch50_tInvMeas100_nGpSub800 --hx -2.0 --eps 10 --delta_t 0.001 --K 9 --J 8 --F 10 --t_synch 50 --t_train 10 --t_invariant_measure 100 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.001/eps10_hx-2.0/tSynch500_tInvMeas2000_nGpSub800 --hx -2.0 --eps 10 --delta_t 0.001 --K 9 --J 8 --F 10 --t_synch 500 --t_train 10 --t_invariant_measure 2000 --n_subsample_gp 800
wait
