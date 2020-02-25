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
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F5_eps-1/Init0 --F 5 --eps -1 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F5_eps-1/Init1 --F 5 --eps -1 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F5_eps-1/Init2 --F 5 --eps -1 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F5_eps-1/Init3 --F 5 --eps -1 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F5_eps-1/Init4 --F 5 --eps -1 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F5_eps-1/Init5 --F 5 --eps -1 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F5_eps-1/Init6 --F 5 --eps -1 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F5_eps-1/Init7 --F 5 --eps -1 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F5_eps-1/Init8 --F 5 --eps -1 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F5_eps-1/Init9 --F 5 --eps -1 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True
wait
