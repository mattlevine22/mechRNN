#!/bin/bash

python3 ../scripts/lorenz63_3dvar_generateData.py --n_trajectories 10 --output_filename TrainData.npz --t_end 50
python3 ../scripts/lorenz63_3dvar_generateData.py --n_trajectories 10 --output_filename TestData.npz --t_end 5

python3 ../scripts/lorenz63_3dvar_train.py --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 0
python3 ../scripts/lorenz63_3dvar_train.py --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 1
python3 ../scripts/lorenz63_3dvar_train.py --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 2
python3 ../scripts/lorenz63_3dvar_train.py --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 3
python3 ../scripts/lorenz63_3dvar_train.py --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 4
python3 ../scripts/lorenz63_3dvar_train.py --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 5
python3 ../scripts/lorenz63_3dvar_train.py --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 6
python3 ../scripts/lorenz63_3dvar_train.py --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 7
python3 ../scripts/lorenz63_3dvar_train.py --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 8
python3 ../scripts/lorenz63_3dvar_train.py --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 9

python3 ../scripts/lorenz63_3dvar_summarize.py --n_train_trajectories 10 --n_test_trajectories 10
