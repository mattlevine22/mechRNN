#INSTRUCTIONS: run each command separately by hand. This script does not have a wait feature.

#0. Characterize loss surfaces (running this suggested using data of length T=100)
sbatch charterize_loss_surface.sh

#1. Generate Data
module purge
module load cuda/9.0
module load python/2.7.15-tf
python3 lorenz63_3dvar_generateData.py --n_trajectories 10 --output_dir /groups/astuart/mlevine/writeup0/l63/3dvar/K_learning/pilot_optimization_run --output_filename TrainData.npz --t_end 100
python3 lorenz63_3dvar_generateData.py --n_trajectories 10 --output_dir /groups/astuart/mlevine/writeup0/l63/3dvar/K_learning/pilot_optimization_run --output_filename TestData.npz --t_end 20

#2. Run GradientDescent learning
sbatch jobs_K_Learning_GradientDescent_FullState.sh
# to test, run:
# python3 l63_3dvar_GradientDescentTrain.py --output_dir test_optimizations --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 0

#3. Summarize all GradientDescent Results
module purge
module load cuda/9.0
module load python/2.7.15-tf
python3 l63_3dvar_GradientDescentTrain_summarize.py --output_dir /groups/astuart/mlevine/writeup0/l63/3dvar/K_learning/pilot_optimization_run --n_trains 3

#4a. Run optimization with NelderMead Full State (can be run concurrently with 4b)
sbatch jobs_K_Learning_NelderMead_FullState.sh
# to test, run:
# python3 l63_3dvar_NelderMeadTrain.py --output_dir test_optimizations --optim_full_state True --n_nelder_inits 2 --max_nelder_sols 10 --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 0

#4b. Run optimization with NelderMead Partial State (can be run concurrently with 4a)
sbatch jobs_K_Learning_NelderMead_PartialState.sh
# to test, run:
# python3 l63_3dvar_NelderMeadTrain.py --output_dir test_optimizations --optim_full_state False --n_nelder_inits 2 --max_nelder_sols 10 --training_data_filename TrainData.npz --testing_data_filename TestData.npz --train_input_index 0

