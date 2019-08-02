from utils import *
import numpy as np
import torch

lr = 0.05 # learning rate
tspan = np.arange(0,1001,1)
yb = 100
c_gamma = 0.05
sim_model_params = (yb, c_gamma)
rnn_model_params = (yb, c_gamma)

sim_model = exp_decay_model
rnn_sim_model = easy_exp_decay_model

n_sims = 3
n_epochs = 1000

train_frac = 0.6
for i in range(n_sims):
	all_dirs = []

	np.random.seed()

	# master output directory name
	output_dir = 'output/experiment6/sim' + str(i+1)
	# simulate clean and noisy data
	input_data, y_clean, y_noisy = make_RNN_data(
	              sim_model, tspan, sim_model_params, noise_frac=0.1, output_dir=output_dir)

	###### do train/test split #######
	n_train = int(np.floor(train_frac*len(y_clean)))
	y_clean_train = y_clean[:n_train]
	y_clean_test = y_clean[n_train:]
	y_noisy_train = y_noisy[:n_train]
	y_noisy_test = y_noisy[n_train:]
	x_train = input_data[:, :n_train]
	x_test = input_data[:, n_train:]

	####### collect normalization information from TRAINING SET ONLY ######
	normz_info_clean = {}
	normz_info_clean['Ymax'] = max(y_clean_train)
	normz_info_clean['Ymin'] = min(y_clean_train)
	# normz_info_clean['Ymean'] = np.mean(y_clean_train)
	# normz_info_clean['Ysd'] = np.std(y_clean_train)
	normz_info_clean['Xmean'] = np.mean(x_train)
	normz_info_clean['Xsd'] = np.std(x_train)

	normz_info_noisy = {}
	normz_info_noisy['Ymax'] = max(y_noisy_train)
	normz_info_noisy['Ymin'] = min(y_noisy_train)
	# normz_info_noisy['Ymean'] = np.mean(y_noisy_train)
	# normz_info_noisy['Ysd'] = np.std(y_noisy_train)
	normz_info_noisy['Xmean'] = np.mean(x_train)
	normz_info_noisy['Xsd'] = np.std(x_train)

	###### normalize TRAINING data #######
	# # y (MIN/MAX [0,1])
	# y_clean_train = (y_clean_train - normz_info_clean['Ymin'])/(normz_info_clean['Ymax'] - normz_info_clean['Ymin'])
	# y_noisy_train = (y_noisy_train - normz_info_noisy['Ymin'])/(normz_info_noisy['Ymax'] - normz_info_noisy['Ymin'])
	# # x (MIN/MAX [0,1])
	# # y_clean_train = (y_clean_train - normz_info_clean['Ymean']) / normz_info_clean['Ysd']
	# # y_noisy_train = (y_noisy_train - normz_info_noisy['Ymean']) / normz_info_noisy['Ysd']
	x_train = (x_train - normz_info_clean['Xmean']) / normz_info_clean['Xsd']

	# ###### normalize TESTING data ########
	# # y (MIN/MAX [0,1])
	# y_clean_test = (y_clean_test - normz_info_clean['Ymin'])/(normz_info_clean['Ymax'] - normz_info_clean['Ymin'])
	# y_noisy_test = (y_noisy_test - normz_info_noisy['Ymin'])/(normz_info_noisy['Ymax'] - normz_info_noisy['Ymin'])
	# x (MIN/MAX [0,1])
	# y_clean_test = (y_clean_test - normz_info_clean['Ymean']) / normz_info_clean['Ysd']
	# y_noisy_test = (y_noisy_test - normz_info_noisy['Ymean']) / normz_info_noisy['Ysd']

	x_test = (x_test - normz_info_clean['Xmean']) / normz_info_clean['Xsd']

	########## NOW start running RNN fits ############

	#### run vanilla RNN ####
	forward = forward_vanilla
	hidden_size = 6

	# train on clean data
	normz_info = normz_info_clean
	run_output_dir = output_dir + '/vanillaRNN_clean'
	all_dirs.append(run_output_dir)
	torch.manual_seed(0)
	y_clean_train_norm = (y_clean_train - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
	y_noisy_train_norm = (y_noisy_train - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
	y_clean_test_norm = (y_clean_test - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
	y_noisy_test_norm = (y_noisy_test - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
	train_RNN(forward,
      y_clean_train_norm, y_clean_train_norm, x_train,
      y_clean_test_norm, y_noisy_test_norm, x_test,
      rnn_model_params, hidden_size, n_epochs, lr,
      run_output_dir, normz_info_clean, rnn_sim_model)

	# train on noisy data
	normz_info = normz_info_noisy
	run_output_dir = output_dir + '/vanillaRNN_noisy'
	all_dirs.append(run_output_dir)
	torch.manual_seed(0)
	y_clean_train_norm = (y_clean_train - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
	y_noisy_train_norm = (y_noisy_train - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
	y_clean_test_norm = (y_clean_test - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
	y_noisy_test_norm = (y_noisy_test - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
	train_RNN(forward,
      y_clean_train_norm, y_noisy_train_norm, x_train,
      y_clean_test_norm, y_noisy_test_norm, x_test,
      rnn_model_params, hidden_size, n_epochs, lr,
      run_output_dir, normz_info, rnn_sim_model)


	#### run mechRNN ###
	forward = forward_mech
	hidden_size = 7

	# train on clean data (random init)
	normz_info = normz_info_clean
	run_output_dir = output_dir + '/mechRNN_clean'
	all_dirs.append(run_output_dir)
	torch.manual_seed(0)
	y_clean_train_norm = (y_clean_train - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
	y_noisy_train_norm = (y_noisy_train - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
	y_clean_test_norm = (y_clean_test - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
	y_noisy_test_norm = (y_noisy_test - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
	train_RNN(forward,
      y_clean_train_norm, y_clean_train_norm, x_train,
      y_clean_test_norm, y_noisy_test_norm, x_test,
      rnn_model_params, hidden_size, n_epochs, lr,
      run_output_dir, normz_info_clean, rnn_sim_model)
	# train on clean data (trivial init)
	run_output_dir = output_dir + '/mechRNN_trivialInit_clean'
	all_dirs.append(run_output_dir)
	torch.manual_seed(0)
	train_RNN(forward,
      y_clean_train_norm, y_clean_train_norm, x_train,
      y_clean_test_norm, y_noisy_test_norm, x_test,
      rnn_model_params, hidden_size, n_epochs, lr,
      run_output_dir, normz_info_clean, rnn_sim_model,
      trivial_init=True)

	# train on noisy data (regular initialization)
	normz_info = normz_info_noisy
	run_output_dir = output_dir + '/mechRNN_noisy'
	all_dirs.append(run_output_dir)
	torch.manual_seed(0)
	y_clean_train_norm = (y_clean_train - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
	y_noisy_train_norm = (y_noisy_train - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
	y_clean_test_norm = (y_clean_test - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
	y_noisy_test_norm = (y_noisy_test - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
	train_RNN(forward,
      y_clean_train_norm, y_noisy_train_norm, x_train,
      y_clean_test_norm, y_noisy_test_norm, x_test,
      rnn_model_params, hidden_size, n_epochs, lr,
      run_output_dir, normz_info_noisy, rnn_sim_model)
	# train on noisy data (trivial initialization)
	run_output_dir = output_dir + '/mechRNN_trivialInit_noisy'
	all_dirs.append(run_output_dir)
	torch.manual_seed(0)
	train_RNN(forward,
      y_clean_train_norm, y_noisy_train_norm, x_train,
      y_clean_test_norm, y_noisy_test_norm, x_test,
      rnn_model_params, hidden_size, n_epochs, lr,
      run_output_dir, normz_info_noisy, rnn_sim_model,
      trivial_init=True)

	#### run mechRNN w/ BAD parameter ###
	forward = forward_mech
	hidden_size = 7

	for badness in [0.01, 0.1, 0.5, 0.9, 0.99]:
		rnn_BAD_model_params = (yb, badness*c_gamma)

		# train on clean data
		normz_info = normz_info_clean
		run_output_dir = output_dir + '/mechRNN_badGamma{}_clean'.format(badness)
		all_dirs.append(run_output_dir)
		torch.manual_seed(0)
		y_clean_train_norm = (y_clean_train - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
		y_noisy_train_norm = (y_noisy_train - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
		y_clean_test_norm = (y_clean_test - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
		y_noisy_test_norm = (y_noisy_test - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
		train_RNN(forward,
	      y_clean_train_norm, y_clean_train_norm, x_train,
	      y_clean_test_norm, y_noisy_test_norm, x_test,
	      rnn_BAD_model_params, hidden_size, n_epochs, lr,
	      run_output_dir, normz_info_clean, rnn_sim_model)

		# train on noisy data
		normz_info = normz_info_noisy
		run_output_dir = output_dir + '/mechRNN_badGamma{}_noisy'.format(badness)
		all_dirs.append(run_output_dir)
		torch.manual_seed(0)
		y_clean_train_norm = (y_clean_train - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
		y_noisy_train_norm = (y_noisy_train - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
		y_clean_test_norm = (y_clean_test - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
		y_noisy_test_norm = (y_noisy_test - normz_info['Ymin'])/(normz_info['Ymax'] - normz_info['Ymin'])
		train_RNN(forward,
	      y_clean_train_norm, y_noisy_train_norm, x_train,
	      y_clean_test_norm, y_noisy_test_norm, x_test,
	      rnn_BAD_model_params, hidden_size, n_epochs, lr,
	      run_output_dir, normz_info_noisy, rnn_sim_model)

	# plot comparative training errors
	compare_fits([d for d in all_dirs if "clean" in d], output_fname=output_dir+'/model_comparisons_clean')
	compare_fits([d for d in all_dirs if "noisy" in d], output_fname=output_dir+'/model_comparisons_noisy')

