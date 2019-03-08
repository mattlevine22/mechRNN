from utils import *
import numpy as np
import torch

lr=0.05 # learning rate
tspan = np.arange(0,1001,1)
yb = 100
c_gamma = 0.05
sim_model_params = (yb, c_gamma)
rnn_model_params = (yb, c_gamma)
rnn_BAD_model_params = (yb, 0.9*c_gamma)

sim_model = exp_decay_model
rnn_sim_model = easy_exp_decay_model

n_sims = 3
n_epochs = 100

train_frac = 0.6
for i in range(n_sims):
	np.random.seed()

	# master output directory name
	output_dir = 'output/experiment2/sim' + str(i+1)
	# simulate clean and noisy data
	input_data, y_clean, y_noisy, normz_info_clean, normz_info_noisy = make_RNN_data(
	              sim_model, tspan, sim_model_params, noise_frac=0.1, output_dir=output_dir)


	# do train/test split
	n_train = int(np.floor(train_frac*len(y_clean)))
	y_clean_train = y_clean[:n_train]
	y_clean_test = y_clean[n_train:]
	y_noisy_train = y_noisy[:n_train]
	y_noisy_test = y_noisy[n_train:]
	x_train = input_data[:,:n_train]
	x_test = input_data[:,n_train:]

	all_dirs = []
	#### run vanilla RNN ####
	forward = forward_vanilla
	hidden_size = 6
	# train on clean data
	run_output_dir = output_dir + '/vanillaRNN_clean'
	all_dirs.append(run_output_dir)
	torch.manual_seed(0)
	train_RNN(forward,
      y_clean_train, y_clean_train, x_train,
      y_clean_test, y_noisy_test, x_test,
      rnn_model_params, hidden_size, n_epochs, lr,
      run_output_dir, normz_info_clean, rnn_sim_model)

	# train on noisy data
	run_output_dir = output_dir + '/vanillaRNN_noisy'
	all_dirs.append(run_output_dir)
	torch.manual_seed(0)
	# train_RNN(forward, input_data, tspan, y_clean, y_noisy,
	  # rnn_model_params, hidden_size, n_epochs, lr, run_output_dir, normz_info_noisy, rnn_sim_model)
	train_RNN(forward,
      y_clean_train, y_noisy_train, x_train,
      y_clean_test, y_noisy_test, x_test,
      rnn_model_params, hidden_size, n_epochs, lr,
      run_output_dir, normz_info_noisy, rnn_sim_model)


	#### run mechRNN ###
	forward = forward_mech
	hidden_size = 7
	# train on clean data
	run_output_dir = output_dir + '/mechRNN_clean'
	all_dirs.append(run_output_dir)
	torch.manual_seed(0)
	# train_RNN(forward, input_data, tspan, y_clean, y_clean,
	#   rnn_model_params, hidden_size, n_epochs, lr, run_output_dir, normz_info_clean, rnn_sim_model)
	train_RNN(forward,
      y_clean_train, y_clean_train, x_train,
      y_clean_test, y_noisy_test, x_test,
      rnn_model_params, hidden_size, n_epochs, lr,
      run_output_dir, normz_info_clean, rnn_sim_model)
	# train on noisy data
	run_output_dir = output_dir + '/mechRNN_noisy'
	all_dirs.append(run_output_dir)
	torch.manual_seed(0)
	# train_RNN(forward, input_data, tspan, y_clean, y_noisy,
	#   rnn_model_params, hidden_size, n_epochs, lr, run_output_dir, normz_info_noisy, rnn_sim_model)
	train_RNN(forward,
      y_clean_train, y_noisy_train, x_train,
      y_clean_test, y_noisy_test, x_test,
      rnn_model_params, hidden_size, n_epochs, lr,
      run_output_dir, normz_info_noisy, rnn_sim_model)

	#### run mechRNN w/ BAD parameter ###
	forward = forward_mech
	hidden_size = 7
	# train on clean data
	run_output_dir = output_dir + '/mechRNN_badParam_clean'
	all_dirs.append(run_output_dir)
	torch.manual_seed(0)
	# train_RNN(forward, input_data, tspan, y_clean, y_clean,
	#   rnn_BAD_model_params, hidden_size, n_epochs, lr, run_output_dir, normz_info_clean, rnn_sim_model)
	train_RNN(forward,
      y_clean_train, y_clean_train, x_train,
      y_clean_test, y_noisy_test, x_test,
      rnn_BAD_model_params, hidden_size, n_epochs, lr,
      run_output_dir, normz_info_clean, rnn_sim_model)
	# train on noisy data
	run_output_dir = output_dir + '/mechRNN_badParam_noisy'
	all_dirs.append(run_output_dir)
	torch.manual_seed(0)
	# train_RNN(forward, input_data, tspan, y_clean, y_noisy,
	#   rnn_BAD_model_params, hidden_size, n_epochs, lr, run_output_dir, normz_info_noisy, rnn_sim_model)
	train_RNN(forward,
      y_clean_train, y_noisy_train, x_train,
      y_clean_test, y_noisy_test, x_test,
      rnn_BAD_model_params, hidden_size, n_epochs, lr,
      run_output_dir, normz_info_noisy, rnn_sim_model)

	# plot comparative training errors
	compare_fits(all_dirs, output_dir)

