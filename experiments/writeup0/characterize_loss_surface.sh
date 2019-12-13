#!/bin/bash

#Submit this script with: sbatch train_submission

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=3   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=100MB   # memory per CPU core
#SBATCH -c 1 # 1 core per task

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load cuda/9.0
module load python/2.7.15-tf

srun --exclusive -N 1 -n 1 python ../scripts/l63_3dvar_characterize_loss_functions.py --savedir /groups/astuart/mlevine/writeup0/l63/3dvar/loss_surface_characterization/n0 &
srun --exclusive -N 1 -n 1 python ../scripts/l63_3dvar_characterize_loss_functions.py --savedir /groups/astuart/mlevine/writeup0/l63/3dvar/loss_surface_characterization/n1 &
srun --exclusive -N 1 -n 1 python ../scripts/l63_3dvar_characterize_loss_functions.py --savedir /groups/astuart/mlevine/writeup0/l63/3dvar/loss_surface_characterization/n2 &
wait