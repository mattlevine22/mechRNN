#!/bin/bash

#SBATCH --time=6:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH -c 1 # 1 core per task

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load python3/3.7.0

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.01/K4_J4_tSynch50_tInvMeas100_nGpSub800 --delta_t 0.01 --K 4 --J 4 --F 50 --t_synch 50 --t_train 10 --t_invariant_measure 100 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.01/K4_J4_tSynch500_tInvMeas100_nGpSub800 --delta_t 0.01 --K 4 --J 4 --F 50 --t_synch 500 --t_train 10 --t_invariant_measure 100 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.01/K4_J4_tSynch500_tInvMeas2000_nGpSub800 --delta_t 0.01 --K 4 --J 4 --F 50 --t_synch 500 --t_train 10 --t_invariant_measure 2000 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.01/K4_J4_tSynch500_tInvMeas2000_nGpSub1600 --delta_t 0.01 --K 4 --J 4 --F 50 --t_synch 500 --t_train 10 --t_invariant_measure 2000 --n_subsample_gp 1600 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.01/tSynch50_tInvMeas100_nGpSub800 --delta_t 0.01 --K 9 --J 8 --F 10 --t_synch 50 --t_train 10 --t_invariant_measure 100 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.01/tSynch500_tInvMeas100_nGpSub800 --delta_t 0.01 --K 9 --J 8 --F 10 --t_synch 500 --t_train 10 --t_invariant_measure 100 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.01/tSynch500_tInvMeas2000_nGpSub800 --delta_t 0.01 --K 9 --J 8 --F 10 --t_synch 500 --t_train 10 --t_invariant_measure 2000 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/reproduce_dima/dt0.01/tSynch500_tInvMeas2000_nGpSub1600 --delta_t 0.01 --K 9 --J 8 --F 10 --t_synch 500 --t_train 10 --t_invariant_measure 2000 --n_subsample_gp 1600
wait
