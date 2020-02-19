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
srun --exclusive -N 1 -n 1 python ../scripts/l96_FullSystem_BadFparameter.py --savedir /groups/astuart/mlevine/writeup0/l96/Fbad_fullSystem_trial0_dt1e-1/Init0 --delta_t 1e-1 --t_train 100 --t_test 20 --n_tests 3 --K 4 --slow_only False --F 10 --fix_seed True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_FullSystem_BadFparameter.py --savedir /groups/astuart/mlevine/writeup0/l96/Fbad_fullSystem_trial0_dt1e-1/Init1 --delta_t 1e-1 --t_train 100 --t_test 20 --n_tests 3 --K 4 --slow_only False --F 10 --fix_seed True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_FullSystem_BadFparameter.py --savedir /groups/astuart/mlevine/writeup0/l96/Fbad_fullSystem_trial0_dt1e-1/Init2 --delta_t 1e-1 --t_train 100 --t_test 20 --n_tests 3 --K 4 --slow_only False --F 10 --fix_seed True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_FullSystem_BadFparameter.py --savedir /groups/astuart/mlevine/writeup0/l96/Fbad_fullSystem_trial0_dt1e-1/Init3 --delta_t 1e-1 --t_train 100 --t_test 20 --n_tests 3 --K 4 --slow_only False --F 10 --fix_seed True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_FullSystem_BadFparameter.py --savedir /groups/astuart/mlevine/writeup0/l96/Fbad_fullSystem_trial0_dt1e-1/Init4 --delta_t 1e-1 --t_train 100 --t_test 20 --n_tests 3 --K 4 --slow_only False --F 10 --fix_seed True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_FullSystem_BadFparameter.py --savedir /groups/astuart/mlevine/writeup0/l96/Fbad_fullSystem_trial0_dt1e-1/Init5 --delta_t 1e-1 --t_train 100 --t_test 20 --n_tests 3 --K 4 --slow_only False --F 10 --fix_seed True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_FullSystem_BadFparameter.py --savedir /groups/astuart/mlevine/writeup0/l96/Fbad_fullSystem_trial0_dt1e-1/Init6 --delta_t 1e-1 --t_train 100 --t_test 20 --n_tests 3 --K 4 --slow_only False --F 10 --fix_seed True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_FullSystem_BadFparameter.py --savedir /groups/astuart/mlevine/writeup0/l96/Fbad_fullSystem_trial0_dt1e-1/Init7 --delta_t 1e-1 --t_train 100 --t_test 20 --n_tests 3 --K 4 --slow_only False --F 10 --fix_seed True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_FullSystem_BadFparameter.py --savedir /groups/astuart/mlevine/writeup0/l96/Fbad_fullSystem_trial0_dt1e-1/Init8 --delta_t 1e-1 --t_train 100 --t_test 20 --n_tests 3 --K 4 --slow_only False --F 10 --fix_seed True &
srun --exclusive -N 1 -n 1 python ../scripts/l96_FullSystem_BadFparameter.py --savedir /groups/astuart/mlevine/writeup0/l96/Fbad_fullSystem_trial0_dt1e-1/Init9 --delta_t 1e-1 --t_train 100 --t_test 20 --n_tests 3 --K 4 --slow_only False --F 10 --fix_seed True
wait

