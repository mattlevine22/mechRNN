#!/bin/bash

module purge
module load python3/3.6.4

# python3 ../scripts/summarize_dt_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_dT_withGP_constantNUMTRAIN_10synchs_10xNdata_TEST

# python3 ../scripts/summarize_dt_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_dT_withGP_constantTIMELENGTH_10synchs_10xNdata_TEST

# python3 ../scripts/summarize_dt_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_dT_withGP_constantNUMTRAIN_10synchs_10xNdata

# python3 ../scripts/summarize_dt_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_dT_withGP_constantTIMELENGTH_10synchs_10xNdata



python3 ../scripts/summarize_epsilon_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_eps_withGP_10synchs

python3 ../scripts/summarize_epsilon_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_eps_withGP_10synchs_TEST

python3 ../scripts/summarize_epsilon_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_eps_withGP_10synchs_TEST_alpha1e-10
python3 ../scripts/summarize_epsilon_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_eps_withGP_10synchs_TEST_alpha1e-8
python3 ../scripts/summarize_epsilon_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_eps_withGP_10synchs_TEST_alpha1e-6
python3 ../scripts/summarize_epsilon_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_eps_withGP_10synchs_TEST_alpha1e-4
python3 ../scripts/summarize_epsilon_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_eps_withGP_10synchs_TEST_alpha1e-2




# python3 ../scripts/summarize_n_data_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_Ndata_withGP_10synchs_TEST

# python3 ../scripts/summarize_n_data_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_Ndata_withGP_continueTrajectory_TEST

# python3 ../scripts/summarize_n_data_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_Ndata_withGP_10synchs

# python3 ../scripts/summarize_n_data_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_Ndata_withGP_continueTrajectory


# python3 ../scripts/summarize_dt_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_dT_withGP_constantNUMTRAIN_10synchs_TEST

# python3 ../scripts/summarize_dt_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_dT_withGP_constantNUMTRAIN_continueTraj_TEST

# python3 ../scripts/summarize_dt_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_dT_withGP_constantTIMELENGTH_10synchs_TEST

# python3 ../scripts/summarize_dt_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_dT_withGP_constantTIMELENGTH_continueTraj_TEST



# python3 ../scripts/summarize_dt_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_dT_withGP_constantNUMTRAIN_10synchs

# python3 ../scripts/summarize_dt_experiments.py ~/mechRNN/experiments/Aug6/lorenz63_dT_withGP_constantTIMELENGTH_10synchs

