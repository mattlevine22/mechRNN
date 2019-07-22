#!/bin/bash

module purge
module load python3/3.6.4

python3 ../scripts/summarize_n_data_experiments.py ~/mechRNN/experiments/July21/lorenz63_Ndata_withGP_10synchs_TEST

python3 ../scripts/summarize_n_data_experiments.py ~/mechRNN/experiments/July21/lorenz63_Ndata_withGP_continueTrajectory_TEST

python3 ../scripts/summarize_n_data_experiments.py ~/mechRNN/experiments/July21/lorenz63_Ndata_withGP_10synchs

python3 ../scripts/summarize_n_data_experiments.py ~/mechRNN/experiments/July21/lorenz63_Ndata_withGP_continueTrajectory
