#!/bin/bash

#Submit this script with: sbatch train_submission

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=10   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=3G   # memory per CPU core
#SBATCH -c 1 # 1 core per task

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load cuda/9.0
module load python/2.7.15-tf



# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init0 --epoch 10000 --n_tests=10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init1 --epoch 10000 --n_tests=10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init2 --epoch 10000 --n_tests=10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init3 --epoch 10000 --n_tests=10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init4 --epoch 10000 --n_tests=10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init5 --epoch 10000 --n_tests=10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init6 --epoch 10000 --n_tests=10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init7 --epoch 10000 --n_tests=10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init8 --epoch 10000 --n_tests=10 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init9 --epoch 10000 --n_tests=10
wait