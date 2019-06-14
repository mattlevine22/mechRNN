#!/bin/bash

#Submit this script with: sbatch train_submission

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=6   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=30G   # memory per CPU core
#SBATCH -c 1 # 1 core per task

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load cuda/9.0
module load python/2.7.15-tf

# Execute jobs in parallel
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps2.py --savedir lorenz63_newTraining/synch_5 --epoch 10000 --delta_t 0.1 --t_train 100 --t_test 100 --t_test_synch 5 --n_tests 10 --n_experiments 2 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps2.py --savedir lorenz63_newTraining/synch_10 --epoch 10000 --delta_t 0.1 --t_train 100 --t_test 100 --t_test_synch 10 --n_tests 10 --n_experiments 2 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps2.py --savedir lorenz63_newTraining/synch_50 --epoch 10000 --delta_t 0.1 --t_train 100 --t_test 100 --t_test_synch 50 --n_tests 10 --n_experiments 2 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps2.py --savedir lorenz63_newTraining/synch_100 --epoch 10000 --delta_t 0.1 --t_train 100 --t_test 100 --t_test_synch 100 --n_tests 10 --n_experiments 2 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps2.py --savedir lorenz63_newTraining/synch_500 --epoch 10000 --delta_t 0.1 --t_train 100 --t_test 100 --t_test_synch 500 --n_tests 10 --n_experiments 2 &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps2.py --savedir lorenz63_newTraining/synch_1000 --epoch 10000 --delta_t 0.1 --t_train 100 --t_test 100 --t_test_synch 1000 --n_tests 10 --n_experiments 2
wait