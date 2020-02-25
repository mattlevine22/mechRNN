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
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F50_eps-3/Init0 --F 50 --eps -3 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F50_eps-3/Init1 --F 50 --eps -3 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F50_eps-3/Init2 --F 50 --eps -3 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F50_eps-3/Init3 --F 50 --eps -3 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F50_eps-3/Init4 --F 50 --eps -3 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F50_eps-3/Init5 --F 50 --eps -3 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F50_eps-3/Init6 --F 50 --eps -3 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F50_eps-3/Init7 --F 50 --eps -3 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F50_eps-3/Init8 --F 50 --eps -3 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_sandbox.py --savedir /groups/astuart/mlevine/writeup0/l96/F50_eps-3/Init9 --F 50 --eps -3 --K 4 --J 4 --delta_t 0.01 --t_train 20 --n_tests 3 --fix_seed True --t_test 20 --t_test_synch 5 --slow_only True --epoch 1000 --ode_int_method RK45 --run_RNN True
wait
