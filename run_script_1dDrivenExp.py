from utils import *
import numpy as np
import torch

lr = 0.05 # learning rate
delta_t = 1
tspan = np.arange(0,1001,delta_t)
(y_equilb, c_gamma) = [100, 0.05]
state_init = [300]
sim_model_params = {'state_names': ['y'], 'state_init':state_init, 'delta_t':delta_t, 'ode_params':(y_equilb, c_gamma)}
rnn_model_params = {'state_names': ['y'], 'state_init':state_init, 'delta_t':delta_t, 'ode_params':(y_equilb, c_gamma)}

sim_model = exp_decay_model
rnn_sim_model = exp_decay_model

drive_system = True

n_sims = 1
n_epochs = 1000

train_frac = 0.6
for i in range(n_sims):
	all_dirs = []

	np.random.seed()

	# master output directory name
	output_dir = '1dDrivenExp_output/experiment2/sim' + str(i+1)
	# simulate clean and noisy data
	input_data, y_clean, y_noisy = make_RNN_data(
	              sim_model, tspan, sim_model_params, noise_frac=0.05, output_dir=output_dir, drive_system=True)

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
	normz_info_clean['Ymax'] = np.max(y_clean_train,axis=0)
	normz_info_clean['Ymin'] = np.min(y_clean_train,axis=0)
	# normz_info_clean['Ymean'] = np.mean(y_clean_train)
	# normz_info_clean['Ysd'] = np.std(y_clean_train)
	normz_info_clean['Xmean'] = np.mean(x_train)
	normz_info_clean['Xsd'] = np.std(x_train)

	normz_info_noisy = {}
	normz_info_noisy['Ymax'] = np.max(y_noisy_train,axis=0)
	normz_info_noisy['Ymin'] = np.min(y_noisy_train,axis=0)
	# normz_info_noisy['Ymean'] = np.mean(y_noisy_train)
	# normz_info_noisy['Ysd'] = np.std(y_noisy_train)
	normz_info_noisy['Xmean'] = np.mean(x_train)
	normz_info_noisy['Xsd'] = np.std(x_train)

	###### MINMAX normalize TRAINING data #######
	# # y (MIN/MAX [0,1])
	# y_clean_train = f_normalize_minmax(normz_info_clean, y_clean_train)
	# y_noisy_train = f_normalize_minmax(normz_info_clean, y_noisy_train)
	# # x (MIN/MAX [0,1])
	# # y_clean_train = (y_clean_train - normz_info_clean['Ymean']) / normz_info_clean['Ysd']
	# # y_noisy_train = (y_noisy_train - normz_info_noisy['Ymean']) / normz_info_noisy['Ysd']
	# x_train = (x_train - normz_info_clean['Xmean']) / normz_info_clean['Xsd']
	x_train_norm = f_normalize_ztrans(normz_info_clean, x_train)

	# ###### normalize TESTING data ########
	# # y (MIN/MAX [0,1])
	# y_clean_test = f_normalize_minmax(normz_info_clean, y_clean_test)
	# y_noisy_test = f_normalize_minmax(normz_info_clean, y_noisy_test)
	# x (MIN/MAX [0,1])
	# y_clean_test = (y_clean_test - normz_info_clean['Ymean']) / normz_info_clean['Ysd']
	# y_noisy_test = (y_noisy_test - normz_info_noisy['Ymean']) / normz_info_noisy['Ysd']

	# x_test = (x_test - normz_info_clean['Xmean']) / normz_info_clean['Xsd']
	x_test_norm = f_normalize_ztrans(normz_info_clean, x_test)

	########## NOW start running RNN fits ############

	#### run vanilla RNN ####
	forward = forward_vanilla

	# train on clean data
	normz_info = normz_info_clean
	for hidden_size in [6]:
		run_output_dir = output_dir + '/vanillaRNN_clean' + '_hs' + str(hidden_size)
		all_dirs.append(run_output_dir)
		torch.manual_seed(0)
		y_clean_train_norm = f_normalize_minmax(normz_info, y_clean_train)
		y_noisy_train_norm = f_normalize_minmax(normz_info, y_noisy_train)
		y_clean_test_norm = f_normalize_minmax(normz_info, y_clean_test)
		y_noisy_test_norm = f_normalize_minmax(normz_info, y_noisy_test)
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
	y_clean_train_norm = f_normalize_minmax(normz_info, y_clean_train)
	y_noisy_train_norm = f_normalize_minmax(normz_info, y_noisy_train)
	y_clean_test_norm = f_normalize_minmax(normz_info, y_clean_test)
	y_noisy_test_norm = f_normalize_minmax(normz_info, y_noisy_test)
	train_RNN(forward,
      y_clean_train_norm, y_noisy_train_norm, x_train,
      y_clean_test_norm, y_noisy_test_norm, x_test,
      rnn_model_params, hidden_size, n_epochs, lr,
      run_output_dir, normz_info, rnn_sim_model)


	#### run mechRNN ###
	forward = forward_mech

	# train on clean data (random init)
	normz_info = normz_info_clean

	# train on clean data (trivial init)
	run_output_dir = output_dir + '/mechRNN_trivialInit_clean'
	all_dirs.append(run_output_dir)
	torch.manual_seed(0)
	y_clean_train_norm = f_normalize_minmax(normz_info, y_clean_train)
	y_noisy_train_norm = f_normalize_minmax(normz_info, y_noisy_train)
	y_clean_test_norm = f_normalize_minmax(normz_info, y_clean_test)
	y_noisy_test_norm = f_normalize_minmax(normz_info, y_noisy_test)
	train_RNN(forward,
      y_clean_train_norm, y_clean_train_norm, x_train,
      y_clean_test_norm, y_noisy_test_norm, x_test,
      rnn_model_params, hidden_size, n_epochs, lr,
      run_output_dir, normz_info_clean, rnn_sim_model,
      trivial_init=True)

	run_output_dir = output_dir + '/mechRNN_clean'
	all_dirs.append(run_output_dir)
	torch.manual_seed(0)
	train_RNN(forward,
      y_clean_train, y_clean_train, x_train,
      y_clean_test, y_noisy_test, x_test,
      rnn_model_params, hidden_size, n_epochs, lr,
      run_output_dir, normz_info_clean, rnn_sim_model)


	# train on noisy data (regular initialization)
	normz_info = normz_info_noisy
	run_output_dir = output_dir + '/mechRNN_noisy'
	all_dirs.append(run_output_dir)
	torch.manual_seed(0)
	y_clean_train_norm = f_normalize_minmax(normz_info, y_clean_train)
	y_noisy_train_norm = f_normalize_minmax(normz_info, y_noisy_train)
	y_clean_test_norm = f_normalize_minmax(normz_info, y_clean_test)
	y_noisy_test_norm = f_normalize_minmax(normz_info, y_noisy_test)
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
	forward = forward_chaos_hybrid_full

	for eps_badness in [0.0, 0.1, 1]:
		rnn_BAD_model_params = {'state_names': ['y'], 'state_init':state_init, 'delta_t':delta_t, 'ode_params':(y_equilb, c_gamma*(1+eps_badness))}

		# train on clean data
		normz_info = normz_info_clean
		run_output_dir = output_dir + '/mechRNN_epsBadness{}_clean'.format(eps_badness)
		all_dirs.append(run_output_dir)
		torch.manual_seed(0)
		y_clean_train_norm = f_normalize_minmax(normz_info, y_clean_train)
		y_noisy_train_norm = f_normalize_minmax(normz_info, y_noisy_train)
		y_clean_test_norm = f_normalize_minmax(normz_info, y_clean_test)
		y_noisy_test_norm = f_normalize_minmax(normz_info, y_noisy_test)
		train_RNN(forward,
	      y_clean_train_norm, y_clean_train_norm, x_train,
	      y_clean_test_norm, y_noisy_test_norm, x_test,
	      rnn_BAD_model_params, hidden_size, n_epochs, lr,
	      run_output_dir, normz_info_clean, rnn_sim_model)

		# train on noisy data
		normz_info = normz_info_noisy
		run_output_dir = output_dir + '/mechRNN_epsBadness{}_noisy'.format(eps_badness)
		all_dirs.append(run_output_dir)
		torch.manual_seed(0)
		y_clean_train_norm = f_normalize_minmax(normz_info, y_clean_train)
		y_noisy_train_norm = f_normalize_minmax(normz_info, y_noisy_train)
		y_clean_test_norm = f_normalize_minmax(normz_info, y_clean_test)
		y_noisy_test_norm = f_normalize_minmax(normz_info, y_noisy_test)
		train_RNN(forward,
	      y_clean_train_norm, y_noisy_train_norm, x_train,
	      y_clean_test_norm, y_noisy_test_norm, x_test,
	      rnn_BAD_model_params, hidden_size, n_epochs, lr,
	      run_output_dir, normz_info_noisy, rnn_sim_model)

	# plot comparative training errors
	compare_fits([d for d in all_dirs if "clean" in d], output_fname=output_dir+'/model_comparisons_clean')
	compare_fits([d for d in all_dirs if "noisy" in d], output_fname=output_dir+'/model_comparisons_noisy')

