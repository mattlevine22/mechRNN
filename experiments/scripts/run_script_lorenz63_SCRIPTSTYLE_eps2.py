from utils import *
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser(description='mechRNN')
parser.add_argument('--epoch', type=int, default=100, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--delta_t', type=float, default=0.1, help='time step of simulation')
parser.add_argument('--t_train', type=float, default=1000, help='length of train simulation')
parser.add_argument('--t_test', type=float, default=1000, help='length of test simulation')
parser.add_argument('--t_test_synch', type=float, default=10, help='length of test simulation')
parser.add_argument('--savedir', type=str, default='default_output', help='parent dir of output')
parser.add_argument('--model_solver', default=lorenz63, help='ode function')
parser.add_argument('--drive_system', type=str2bool, default=False, help='whether to force the system with a time-dependent driver')
parser.add_argument('--n_tests', type=int, default=1, help='number of independent testing sets to use')
parser.add_argument('--n_experiments', type=int, default=1, help='number of sim/fitting experiments to do')
FLAGS = parser.parse_args()


def main():
	(a,b,c) = [10, 28, 8/3] #chaotic lorenz parameters
	state_init = [] #[-5, 0, 30]

	lr = FLAGS.lr # learning rate
	delta_t = FLAGS.delta_t #0.01
	tspan_train = np.arange(0,FLAGS.t_train,delta_t)  #np.arange(0,10000,delta_t)
	tspan_test = np.arange(0,(FLAGS.t_test_synch+FLAGS.t_test),delta_t)  #np.arange(0,10000,delta_t)
	ntsynch = int(FLAGS.t_test_synch/delta_t)
	sim_model = FLAGS.model_solver
	rnn_sim_model = FLAGS.model_solver

	drive_system = FLAGS.drive_system #False

	n_epochs = FLAGS.epoch #1

	sim_model_params = {'state_names': ['x','y','z'], 'state_init':state_init, 'delta_t':delta_t, 'ode_params':(a, b, c)}
	rnn_model_params = {'state_names': ['x','y','z'], 'state_init':state_init, 'delta_t':delta_t, 'ode_params':(a, b, c)}
	all_dirs = []

	np.random.seed()

	# master output directory name
	output_dir = FLAGS.savedir + '_output'
	# simulate clean and noisy data
	(input_data_train, y_clean_train, y_noisy_train,
	y_clean_test_vec, y_noisy_test_vec, x_test_vec) = make_RNN_data2(
	              sim_model, tspan_train, tspan_test, sim_model_params,
	              noise_frac=0.05, output_dir=output_dir, drive_system=False,
	              n_test_sets=FLAGS.n_tests,
	              f_get_state_inits=get_lorenz_inits)

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

	normz_info = normz_info_clean
	y_clean_train_norm = f_normalize_minmax(normz_info,y_clean_train)
	y_noisy_train_norm = f_normalize_minmax(normz_info,y_noisy_train)
	y_clean_test_vec_norm = np.copy(y_clean_test_vec[:,ntsynch:,:])
	y_noisy_test_vec_norm = np.copy(y_noisy_test_vec[:,ntsynch:,:])
	y_clean_testSynch_vec_norm = np.copy(y_clean_test_vec[:,:ntsynch,:])
	y_noisy_testSynch_vec_norm = np.copy(y_noisy_test_vec[:,:ntsynch,:])
	for k in range(FLAGS.n_tests):
		y_clean_testSynch_vec_norm[k,:,:] = f_normalize_minmax(normz_info, y_clean_test_vec[k,:ntsynch,:])
		y_noisy_testSynch_vec_norm[k,:,:] = f_normalize_minmax(normz_info, y_noisy_test_vec[k,:ntsynch,:])
		y_clean_test_vec_norm[k,:,:] = f_normalize_minmax(normz_info, y_clean_test_vec[k,ntsynch:,:])
		y_noisy_test_vec_norm[k,:,:] = f_normalize_minmax(normz_info, y_noisy_test_vec[k,ntsynch:,:])


	########## NOW start running RNN fits ############
	for n in range(FLAGS.n_experiments):
		for hidden_size in [50]:
			#### run vanilla RNN ####
			forward = forward_chaos_pureML
			# train on clean data
			run_output_dir = output_dir + '_iter{0}'.format(n) + '/vanillaRNN_clean_hs{0}'.format(hidden_size)
			all_dirs.append(run_output_dir)
			# torch.manual_seed(0)
			train_chaosRNN(forward,
		      y_clean_train_norm, y_clean_train_norm,
		      y_clean_test_vec_norm, y_noisy_test_vec_norm,
		      y_clean_testSynch_vec_norm, y_noisy_testSynch_vec_norm,
		      rnn_model_params, hidden_size, n_epochs, lr,
		      run_output_dir, normz_info_clean, rnn_sim_model,
		      stack_hidden=False, stack_output=False)

			# # train on noisy data
			# normz_info = normz_info_noisy
			# (y_clean_train_norm, y_noisy_train_norm,
			# 	y_clean_test_vec_norm, y_noisy_test_vec_norm) = [
			# 		f_normalize_minmax(normz_info, y) for y in y_list]
			# (y_clean_train_norm, y_noisy_train_norm,
			# 	y_clean_test_vec_norm, y_noisy_test_vec_norm) = [
			# 		f_normalize_minmax(normz_info, y) for y in y_list]
			# run_output_dir = output_dir + '/vanillaRNN_noisy_hs{0}'.format(hidden_size)
			# all_dirs.append(run_output_dir)
			# torch.manual_seed(0)
			# train_chaosRNN(forward,
		 #      y_clean_train_norm, y_noisy_train_norm,
		 #      y_clean_test_vec_norm, y_noisy_test_vec_norm,
		 #      rnn_model_params, hidden_size, n_epochs, lr,
		 #      run_output_dir, normz_info, rnn_sim_model,
		 #      stack_hidden=False, stack_output=False)


			#### run mechRNN ###
			# forward = forward_chaos_hybrid_full
			# # train on clean data (random init)
			# run_output_dir = output_dir + '/mechRNN_clean_hs{0}'.format(hidden_size)
			# all_dirs.append(run_output_dir)
			# torch.manual_seed(0)
			# train_chaosRNN(forward,
		 #      y_clean_train_norm, y_clean_train_norm,
		 #      y_clean_test_vec_norm, y_noisy_test_vec_norm,
		 #      y_clean_testSynch_vec_norm, y_noisy_testSynch_vec_norm,
		 #      rnn_model_params, hidden_size, n_epochs, lr,
		 #      run_output_dir, normz_info_clean, rnn_sim_model)


		# train on noisy data (regular initialization)
		# normz_info = normz_info_noisy
		# (y_clean_train_norm, y_noisy_train_norm,
		# 	y_clean_test_vec_norm, y_noisy_test_vec_norm) = [
		# 		f_normalize_minmax(normz_info, y) for y in y_list]
		# run_output_dir = output_dir + '/mechRNN_noisy'
		# all_dirs.append(run_output_dir)
		# torch.manual_seed(0)
		# train_chaosRNN(forward,
	 #      y_clean_train_norm, y_noisy_train_norm,
	 #      y_clean_test_vec_norm, y_noisy_test_vec_norm,
	 #      rnn_model_params, hidden_size, n_epochs, lr,
	 #      run_output_dir, normz_info_noisy, rnn_sim_model)
		# # train on noisy data (trivial initialization)
		# run_output_dir = output_dir + '/mechRNN_trivialInit_noisy'
		## all_dirs.append(run_output_dir)
		# torch.manual_seed(0)
		# train_chaosRNN(forward,
	 #      y_clean_train_norm, y_noisy_train_norm,
	 #      y_clean_test_vec_norm, y_noisy_test_vec_norm,
	 #      rnn_model_params, hidden_size, max(1,int(n_epochs/10)), lr,
	 #      run_output_dir, normz_info_noisy, rnn_sim_model,
	 #      trivial_init=True)

		#### run mechRNN w/ BAD parameter ###
		forward = forward_chaos_hybrid_full

		for eps_badness in [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.4]:
			rnn_BAD_model_params = {'state_names': ['x','y','z'], 'state_init':state_init, 'delta_t':delta_t, 'ode_params':(a, b*(1+eps_badness), c)}

			run_output_dir = output_dir + '_iter{0}'.format(n) + '/mechRNN_epsBadness{0}_clean_hs{1}'.format(eps_badness, hidden_size)
			all_dirs.append(run_output_dir)
			# torch.manual_seed(0)
			train_chaosRNN(forward,
		      y_clean_train_norm, y_clean_train_norm,
		      y_clean_test_vec_norm, y_noisy_test_vec_norm,
	  	      y_clean_testSynch_vec_norm, y_noisy_testSynch_vec_norm,
		      rnn_BAD_model_params, hidden_size, n_epochs, lr,
		      run_output_dir, normz_info_clean, rnn_sim_model)

			# train on noisy data
			# normz_info = normz_info_noisy
			# (y_clean_train_norm, y_noisy_train_norm,
			# 	y_clean_test_vec_norm, y_noisy_test_vec_norm) = [
			# 		f_normalize_minmax(normz_info, y) for y in y_list]
			# run_output_dir = output_dir + '/mechRNN_epsBadness{0}_noisy_hs{1}'.format(eps_badness, hidden_size)
			# all_dirs.append(run_output_dir)
			# torch.manual_seed(0)
			# train_chaosRNN(forward,
		 #      y_clean_train_norm, y_noisy_train_norm,
		 #      y_clean_test_vec_norm, y_noisy_test_vec_norm,
		 #      rnn_BAD_model_params, hidden_size, n_epochs, lr,
		 #      run_output_dir, normz_info_noisy, rnn_sim_model)

		# plot comparative training errors
		compare_fits([d for d in all_dirs if "clean" in d], output_fname=output_dir+'/model_comparisons_clean')
		# compare_fits([d for d in all_dirs if "noisy" in d], output_fname=output_dir+'/model_comparisons_noisy')

if __name__ == '__main__':
	main()

