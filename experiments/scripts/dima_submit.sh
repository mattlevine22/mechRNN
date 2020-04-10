#!/bin/bash

#SBATCH --time=6:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH -c 1 # 1 core per task

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load python3/3.7.0

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/l96_dt_trials_v2/dt0.001/F50_eps0.0078125/reproduce_dima_tSynch50_tInvMeas100_nGpSub800 --t_synch 50 --t_train 10 --t_invariant_measure 100 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/l96_dt_trials_v2/dt0.001/F50_eps0.0078125/reproduce_dima_tSynch500_tInvMeas100_nGpSub800 --t_synch 500 --t_train 10 --t_invariant_measure 100 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/l96_dt_trials_v2/dt0.001/F50_eps0.0078125/reproduce_dima_tSynch500_tInvMeas2000_nGpSub800 --t_synch 500 --t_train 10 --t_invariant_measure 2000 --n_subsample_gp 800 &
srun --exclusive -N 1 -n 1 python3 reproduce_dima.py --output_dir /groups/astuart/mlevine/writeup0/l96_dt_trials_v2/dt0.001/F50_eps0.0078125/reproduce_dima_tSynch500_tInvMeas1000_nGpSub800 --t_synch 500 --t_train 10 --t_invariant_measure 2000 --n_subsample_gp 5000 &
wait
