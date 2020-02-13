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
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/gp_flow_learning_dt1e-1/Init0 --epoch 2 --delta_t 1e-1 --t_train 100 --n_tests 10 --fix_seed True --ode_int_atol 1.5e-8 --ode_int_rtol 1.5e-8 --ode_int_method 'RK45' &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/gp_flow_learning_dt1e-1/Init1 --epoch 2 --delta_t 1e-1 --t_train 100 --n_tests 10 --fix_seed True --ode_int_atol 1.5e-8 --ode_int_rtol 1.5e-8 --ode_int_method 'RK45' &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/gp_flow_learning_dt1e-1/Init2 --epoch 2 --delta_t 1e-1 --t_train 100 --n_tests 10 --fix_seed True --ode_int_atol 1.5e-8 --ode_int_rtol 1.5e-8 --ode_int_method 'RK45' &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/gp_flow_learning_dt1e-1/Init3 --epoch 2 --delta_t 1e-1 --t_train 100 --n_tests 10 --fix_seed True --ode_int_atol 1.5e-8 --ode_int_rtol 1.5e-8 --ode_int_method 'RK45' &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/gp_flow_learning_dt1e-1/Init4 --epoch 2 --delta_t 1e-1 --t_train 100 --n_tests 10 --fix_seed True --ode_int_atol 1.5e-8 --ode_int_rtol 1.5e-8 --ode_int_method 'RK45' &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/gp_flow_learning_dt1e-1/Init5 --epoch 2 --delta_t 1e-1 --t_train 100 --n_tests 10 --fix_seed True --ode_int_atol 1.5e-8 --ode_int_rtol 1.5e-8 --ode_int_method 'RK45' &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/gp_flow_learning_dt1e-1/Init6 --epoch 2 --delta_t 1e-1 --t_train 100 --n_tests 10 --fix_seed True --ode_int_atol 1.5e-8 --ode_int_rtol 1.5e-8 --ode_int_method 'RK45' &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/gp_flow_learning_dt1e-1/Init7 --epoch 2 --delta_t 1e-1 --t_train 100 --n_tests 10 --fix_seed True --ode_int_atol 1.5e-8 --ode_int_rtol 1.5e-8 --ode_int_method 'RK45' &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/gp_flow_learning_dt1e-1/Init8 --epoch 2 --delta_t 1e-1 --t_train 100 --n_tests 10 --fix_seed True --ode_int_atol 1.5e-8 --ode_int_rtol 1.5e-8 --ode_int_method 'RK45' &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_eps_withRNNresiduals.py --savedir /groups/astuart/mlevine/writeup0/l63/gp_flow_learning_dt1e-1/Init9 --epoch 2 --delta_t 1e-1 --t_train 100 --n_tests 10 --fix_seed True --ode_int_atol 1.5e-8 --ode_int_rtol 1.5e-8 --ode_int_method 'RK45'
wait

### Need to set fix_seed=True and check on definition of residual RNN