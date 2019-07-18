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
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init0 --continue_trajectory True --epoch 10000 --delta_t 0.1 --t_test 50 --n_experiments 1 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init1 --continue_trajectory True --epoch 10000 --delta_t 0.1 --t_test 50 --n_experiments 1 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init2 --continue_trajectory True --epoch 10000 --delta_t 0.1 --t_test 50 --n_experiments 1 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init3 --continue_trajectory True --epoch 10000 --delta_t 0.1 --t_test 50 --n_experiments 1 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init4 --continue_trajectory True --epoch 10000 --delta_t 0.1 --t_test 50 --n_experiments 1 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init5 --continue_trajectory True --epoch 10000 --delta_t 0.1 --t_test 50 --n_experiments 1 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init6 --continue_trajectory True --epoch 10000 --delta_t 0.1 --t_test 50 --n_experiments 1 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init7 --continue_trajectory True --epoch 10000 --delta_t 0.1 --t_test 50 --n_experiments 1 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init8 --continue_trajectory True --epoch 10000 --delta_t 0.1 --t_test 50 --n_experiments 1 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_Ndata.py --savedir lorenz63_Ndata_withGP_10synchs/Init9 --continue_trajectory True --epoch 10000 --delta_t 0.1 --t_test 50 --n_experiments 1 --random_state_inits True --compute_kl False
wait