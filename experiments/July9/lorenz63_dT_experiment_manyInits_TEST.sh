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
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_TEST/Init0 --epoch 1000 --n_sim_points 3000 --train_frac 0.33 --n_experiments 1 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_TEST/Init1 --epoch 1000 --n_sim_points 3000 --train_frac 0.33 --n_experiments 1 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_TEST/Init2 --epoch 1000 --n_sim_points 3000 --train_frac 0.33 --n_experiments 1 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_TEST/Init3 --epoch 1000 --n_sim_points 3000 --train_frac 0.33 --n_experiments 1 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_TEST/Init4 --epoch 1000 --n_sim_points 3000 --train_frac 0.33 --n_experiments 1 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_TEST/Init5 --epoch 1000 --n_sim_points 3000 --train_frac 0.33 --n_experiments 1 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_TEST/Init6 --epoch 1000 --n_sim_points 3000 --train_frac 0.33 --n_experiments 1 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_TEST/Init7 --epoch 1000 --n_sim_points 3000 --train_frac 0.33 --n_experiments 1 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_TEST/Init8 --epoch 1000 --n_sim_points 3000 --train_frac 0.33 --n_experiments 1 --random_state_inits True --compute_kl False &
srun --exclusive -N 1 -n 1 python ../scripts/run_script_lorenz63_SCRIPTSTYLE_dT.py --savedir lorenz63_dT_withGP_TEST/Init9 --epoch 1000 --n_sim_points 3000 --train_frac 0.33 --n_experiments 1 --random_state_inits True --compute_kl False
wait