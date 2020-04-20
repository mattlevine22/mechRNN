#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH -c 1 # 1 core per task

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load python3/3.7.0

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.001/eps0.008/tSynch50_tInvMeas100_nGpSub800 --eps 0.008 --delta_t 0.001 --K 9 --J 8 --F 10 --t_synch 50 --t_train 10 --t_invariant_measure 100 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.001/eps0.008/tSynch500_tInvMeas2000_nGpSub800 --eps 0.008 --delta_t 0.001 --K 9 --J 8 --F 10 --t_synch 500 --t_train 10 --t_invariant_measure 2000 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.001/eps0.01/tSynch50_tInvMeas100_nGpSub800 --eps 0.01 --delta_t 0.001 --K 9 --J 8 --F 10 --t_synch 50 --t_train 10 --t_invariant_measure 100 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.001/eps0.01/tSynch500_tInvMeas2000_nGpSub800 --eps 0.01 --delta_t 0.001 --K 9 --J 8 --F 10 --t_synch 500 --t_train 10 --t_invariant_measure 2000 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.001/eps0.05/tSynch50_tInvMeas100_nGpSub800 --eps 0.05 --delta_t 0.001 --K 9 --J 8 --F 10 --t_synch 50 --t_train 10 --t_invariant_measure 100 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.001/eps0.05/tSynch500_tInvMeas2000_nGpSub800 --eps 0.05 --delta_t 0.001 --K 9 --J 8 --F 10 --t_synch 500 --t_train 10 --t_invariant_measure 2000 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.001/eps0.1/tSynch50_tInvMeas100_nGpSub800 --eps 0.1 --delta_t 0.001 --K 9 --J 8 --F 10 --t_synch 50 --t_train 10 --t_invariant_measure 100 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.001/eps0.1/tSynch500_tInvMeas2000_nGpSub800 --eps 0.1 --delta_t 0.001 --K 9 --J 8 --F 10 --t_synch 500 --t_train 10 --t_invariant_measure 2000 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.001/eps0.5/tSynch50_tInvMeas100_nGpSub800 --eps 0.5 --delta_t 0.001 --K 9 --J 8 --F 10 --t_synch 50 --t_train 10 --t_invariant_measure 100 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.001/eps0.5/tSynch500_tInvMeas2000_nGpSub800 --eps 0.5 --delta_t 0.001 --K 9 --J 8 --F 10 --t_synch 500 --t_train 10 --t_invariant_measure 2000 --n_subsample_gp 800
wait
