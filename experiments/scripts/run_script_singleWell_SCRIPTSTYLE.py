from utils import *
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser(description='mechRNN')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--delta_t', type=float, default=0.01, help='time step of simulation')
parser.add_argument('--t_end', type=float, default=1000, help='length of simulation')
parser.add_argument('--train_frac', type=float, default=0.6, help='fraction of simulated data for training')
parser.add_argument('--savedir', type=str, default='default_output', help='parent dir of output')
parser.add_argument('--model_solver', default=double_well, help='ode function')
parser.add_argument('--drive_system', type=str2bool, default=False, help='whether to force the system with a time-dependent driver')
parser.add_argument('--n_experiments', type=int, default=1, help='number of sim/fitting experiments to do')
parser.add_argument(' ', type=int, default=1, help='placeholder')
FLAGS = parser.parse_args()

def main():

	(a, b, c) = [-1, 0, 0]
	my_state_inits = [[3]]

	lr = FLAGS.lr # learning rate
	delta_t = FLAGS.delta_t #0.01
	tspan = np.arange(0,FLAGS.t_end,delta_t)  #np.arange(0,10000,delta_t)
	sim_model = FLAGS.model_solver
	rnn_sim_model = FLAGS.model_solver

	drive_system = FLAGS.drive_system #False

	n_sims = FLAGS.n_experiments #1
	n_epochs = FLAGS.epoch #1

	train_frac = FLAGS.train_frac #0.9995
	i = 0
	for state_init in [[1,0]]:
		i += 1
		sim_model_params = {'state_names': ['x','y'], 'state_init':state_init, 'delta_t':delta_t, 'ode_params':(a, b, c)}
		rnn_model_params = {'state_names': ['x','y'], 'state_init':state_init, 'delta_t':delta_t, 'ode_params':(a, b, c)}
		all_dirs = []

		np.random.seed()

		# master output directory name
		output_dir = FLAGS.savedir + '_' + str(i+1)

		# simulate clean and noisy data
		input_data, y_clean, y_noisy = make_RNN_data(
		              sim_model, tspan, sim_model_params, noise_frac=0.05, output_dir=output_dir, drive_system=False)

		###### do train/test split #######
		n_train = int(np.floor(train_frac*len(y_clean)))
		y_clean_train = y_clean[:n_train]
		y_clean_test = y_clean[n_train:]
		y_noisy_train = y_noisy[:n_train]
		y_noisy_test = y_noisy[n_train:]
		# x_train = input_data[:, :n_train]
		# x_test = input_data[:, n_train:]
		y_list = [y_clean_train, y_noisy_train, y_clean_test, y_noisy_test]

		####### collect normalization information from TRAINING SET ONLY ######
		normz_info_clean = {}
		normz_info_clean['Ymax'] = np.max(y_clean_train,axis=0)
		normz_info_clean['Ymin'] = np.min(y_clean_train,axis=0)
		normz_info_clean['Ymean'] = np.mean(y_clean_train)
		normz_info_clean['Ysd'] = np.std(y_clean_train)
		# normz_info_clean['Xmean'] = np.mean(x_train)
		# normz_info_clean['Xsd'] = np.std(x_train)

		normz_info_noisy = {}
		normz_info_noisy['Ymax'] = np.max(y_noisy_train,axis=0)
		normz_info_noisy['Ymin'] = np.min(y_noisy_train,axis=0)
		normz_info_noisy['Ymean'] = np.mean(y_noisy_train)
		normz_info_noisy['Ysd'] = np.std(y_noisy_train)
		# normz_info_noisy['Xmean'] = np.mean(x_train)
		# normz_info_noisy['Xsd'] = np.std(x_train)

		###### MINMAX normalize TRAINING data #######
		# # y (MIN/MAX [0,1])
		# y_clean_train = f_normalize_minmax(normz_info_clean, y_clean_train)
		# y_noisy_train = f_normalize_minmax(normz_info_clean, y_noisy_train)
		# # x (MIN/MAX [0,1])
		# # y_clean_train = (y_clean_train - normz_info_clean['Ymean']) / normz_info_clean['Ysd']
		# # y_noisy_train = (y_noisy_train - normz_info_noisy['Ymean']) / normz_info_noisy['Ysd']
		# x_train = (x_train - normz_info_clean['Xmean']) / normz_info_clean['Xsd']

		# ###### normalize TESTING data ########
		# # y (MIN/MAX [0,1])
		# y_clean_test = f_normalize_minmax(normz_info_clean, y_clean_test)
		# y_noisy_test = f_normalize_minmax(normz_info_clean, y_noisy_test)
		# x (MIN/MAX [0,1])
		# y_clean_test = (y_clean_test - normz_info_clean['Ymean']) / normz_info_clean['Ysd']
		# y_noisy_test = (y_noisy_test - normz_info_noisy['Ymean']) / normz_info_noisy['Ysd']

		# x_test = (x_test - normz_info_clean['Xmean']) / normz_info_clean['Xsd']

		########## NOW start running RNN fits ############

		#### run vanilla RNN ####
		forward = forward_chaos_pureML
		for hidden_size in [7,8,9]:
			# train on clean data
			normz_info = normz_info_clean
			(y_clean_train_norm, y_noisy_train_norm,
				y_clean_test_norm, y_noisy_test_norm) = [
					f_normalize_minmax(normz_info, y) for y in y_list]
			run_output_dir = output_dir + '/vanillaRNN_clean' + '_hs' + str(hidden_size)
			all_dirs.append(run_output_dir)
			torch.manual_seed(0)
			train_chaosRNN(forward,
		      y_clean_train_norm, y_clean_train_norm,
		      y_clean_test_norm, y_noisy_test_norm,
		      rnn_model_params, hidden_size, n_epochs, lr,
		      run_output_dir, normz_info, rnn_sim_model,
		      stack_hidden=False, stack_output=False)

			# # train on noisy data
			# normz_info = normz_info_noisy
			# (y_clean_train_norm, y_noisy_train_norm,
			# 	y_clean_test_norm, y_noisy_test_norm) = [
			# 		f_normalize_minmax(normz_info, y) for y in y_list]
			# (y_clean_train_norm, y_noisy_train_norm,
			# 	y_clean_test_norm, y_noisy_test_norm) = [
			# 		f_normalize_minmax(normz_info, y) for y in y_list]
			# run_output_dir = output_dir + '/vanillaRNN_noisy' + '_hs' + str(hidden_size)
			# all_dirs.append(run_output_dir)
			# torch.manual_seed(0)
			# train_chaosRNN(forward,
		 #      y_clean_train_norm, y_noisy_train_norm,
		 #      y_clean_test_norm, y_noisy_test_norm,
		 #      rnn_model_params, hidden_size, n_epochs, lr,
		 #      run_output_dir, normz_info, rnn_sim_model,
		 #      stack_hidden=False, stack_output=False)


		#### run mechRNN ###
		forward = forward_chaos_hybrid_full

		# train on clean data (random init)
		normz_info = normz_info_clean
		(y_clean_train_norm, y_noisy_train_norm,
			y_clean_test_norm, y_noisy_test_norm) = [
				f_normalize_minmax(normz_info, y) for y in y_list]

		# train on clean data (trivial init)
		run_output_dir = output_dir + '/mechRNN_trivialInit_clean'
		all_dirs.append(run_output_dir)
		torch.manual_seed(0)
		train_chaosRNN(forward,
	      y_clean_train_norm, y_clean_train_norm,
	      y_clean_test_norm, y_noisy_test_norm,
	      rnn_model_params, hidden_size, n_epochs, lr,
	      run_output_dir, normz_info_clean, rnn_sim_model,
	      trivial_init=True)

		run_output_dir = output_dir + '/mechRNN_clean'
		all_dirs.append(run_output_dir)
		torch.manual_seed(0)
		train_chaosRNN(forward,
	      y_clean_train_norm, y_clean_train_norm,
	      y_clean_test_norm, y_noisy_test_norm,
	      rnn_model_params, hidden_size, n_epochs, lr,
	      run_output_dir, normz_info_clean, rnn_sim_model)


		# train on noisy data (regular initialization)
		# normz_info = normz_info_noisy
		# (y_clean_train_norm, y_noisy_train_norm,
		# 	y_clean_test_norm, y_noisy_test_norm) = [
		# 		f_normalize_minmax(normz_info, y) for y in y_list]
		# run_output_dir = output_dir + '/mechRNN_noisy'
		# all_dirs.append(run_output_dir)
		# torch.manual_seed(0)
		# train_chaosRNN(forward,
	 #      y_clean_train_norm, y_noisy_train_norm,
	 #      y_clean_test_norm, y_noisy_test_norm,
	 #      rnn_model_params, hidden_size, n_epochs, lr,
	 #      run_output_dir, normz_info_noisy, rnn_sim_model)
		# # train on noisy data (trivial initialization)
		# run_output_dir = output_dir + '/mechRNN_trivialInit_noisy'
		# all_dirs.append(run_output_dir)
		# torch.manual_seed(0)
		# train_chaosRNN(forward,
	 #      y_clean_train_norm, y_noisy_train_norm,
	 #      y_clean_test_norm, y_noisy_test_norm,
	 #      rnn_model_params, hidden_size, n_epochs, lr,
	 #      run_output_dir, normz_info_noisy, rnn_sim_model,
	 #      trivial_init=True)

		#### run mechRNN w/ BAD parameter ###
		forward = forward_chaos_hybrid_full

		for eps_badness in [0.05, 1, 10]:
			rnn_BAD_model_params = {'state_names': ['x','y'], 'state_init':state_init, 'delta_t':delta_t, 'ode_params':(a, b*(1+eps_badness), c)}

			# train on clean data
			normz_info = normz_info_clean
			(y_clean_train_norm, y_noisy_train_norm,
				y_clean_test_norm, y_noisy_test_norm) = [
					f_normalize_minmax(normz_info, y) for y in y_list]
			run_output_dir = output_dir + '/mechRNN_epsBadness{}_clean'.format(eps_badness)
			all_dirs.append(run_output_dir)
			torch.manual_seed(0)
			train_chaosRNN(forward,
		      y_clean_train_norm, y_clean_train_norm,
		      y_clean_test_norm, y_noisy_test_norm,
		      rnn_BAD_model_params, hidden_size, n_epochs, lr,
		      run_output_dir, normz_info_clean, rnn_sim_model)

			# train on noisy data
			# normz_info = normz_info_noisy
			# (y_clean_train_norm, y_noisy_train_norm,
			# 	y_clean_test_norm, y_noisy_test_norm) = [
			# 		f_normalize_minmax(normz_info, y) for y in y_list]
			# run_output_dir = output_dir + '/mechRNN_epsBadness{}_noisy'.format(eps_badness)
			# all_dirs.append(run_output_dir)
			# torch.manual_seed(0)
			# train_chaosRNN(forward,
		 #      y_clean_train_norm, y_noisy_train_norm,
		 #      y_clean_test_norm, y_noisy_test_norm,
		 #      rnn_BAD_model_params, hidden_size, n_epochs, lr,
		 #      run_output_dir, normz_info_noisy, rnn_sim_model)

		# plot comparative training errors
		compare_fits([d for d in all_dirs if "clean" in d], output_fname=output_dir+'/model_comparisons_clean')
		# compare_fits([d for d in all_dirs if "noisy" in d], output_fname=output_dir+'/model_comparisons_noisy')

