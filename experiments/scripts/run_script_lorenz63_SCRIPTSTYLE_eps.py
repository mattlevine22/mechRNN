from utils import *
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser(description='mechRNN')
parser.add_argument('--epoch', type=int, default=4, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--delta_t', type=float, default=0.1, help='time step of simulation')
parser.add_argument('--t_train', type=float, default=200, help='length of train simulation')
parser.add_argument('--t_test', type=float, default=100, help='length of test simulation')
parser.add_argument('--t_test_synch', type=float, default=10, help='length of test simulation')
parser.add_argument('--savedir', type=str, default='default_output', help='parent dir of output')
parser.add_argument('--model_solver', default=lorenz63, help='ode function')
parser.add_argument('--drive_system', type=str2bool, default=False, help='whether to force the system with a time-dependent driver')
parser.add_argument('--n_tests', type=int, default=1, help='number of independent testing sets to use')
parser.add_argument('--n_experiments', type=int, default=1, help='number of sim/fitting experiments to do')
parser.add_argument('--random_state_inits', type=str2bool, default=True, help='whether to randomly initialize initial conditions for simulated trajectory')
parser.add_argument('--compute_kl', type=str2bool, default=False, help='whether to compute KL divergence between test set density and predicted density')
parser.add_argument('--continue_trajectory', type=str2bool, default=False, help='if true, ignore n_tests and synch length. Instead, simply continue train trajectory into the test trajectory.')
FLAGS = parser.parse_args()


def main():
	(a,b,c) = [10, 28, 8/3]
	if FLAGS.random_state_inits:
		# sampel uniform initial conditions
		(xmin, xmax) = (-10,10)
		(ymin, ymax) = (-20,30)
		(zmin, zmax) = (10,40)

		xrand = xmin+(xmax-xmin)*np.random.random()
		yrand = ymin+(ymax-ymin)*np.random.random()
		zrand = zmin+(zmax-zmin)*np.random.random()
		my_state_inits = [[xrand, yrand, zrand]]
	else:
		my_state_inits = [[-5, 0, 30]]


	if FLAGS.continue_trajectory:
		FLAGS.n_tests = 1
		FLAGS.t_test_synch = 0
	else:
		if FLAGS.t_test_synch < FLAGS.delta_t:
			raise ValueError('t_test_synch (synch-length) must be larger than delta_t step size')

	lr = FLAGS.lr # learning rate
	delta_t = FLAGS.delta_t #0.01
	tspan_train = np.arange(0,FLAGS.t_train,delta_t)  #np.arange(0,10000,delta_t)
	tspan_test = np.arange(0,(FLAGS.t_test_synch+FLAGS.t_test),delta_t)  #np.arange(0,10000,delta_t)
	ntsynch = int(FLAGS.t_test_synch/delta_t)
	sim_model = FLAGS.model_solver
	rnn_sim_model = FLAGS.model_solver

	drive_system = FLAGS.drive_system #False

	n_epochs = FLAGS.epoch #1

	# train_frac = FLAGS.train_frac #0.9995
	i = 0
	for state_init in my_state_inits:
		i += 1
		sim_model_params = {'state_names': ['x','y','z'], 'state_init':state_init, 'delta_t':delta_t, 'ode_params':(a, b, c)}
		rnn_model_params = {'state_names': ['x','y','z'], 'state_init':state_init, 'delta_t':delta_t, 'ode_params':(a, b, c)}
		all_dirs = []

		# np.random.seed()

		# master output directory name
		output_dir = FLAGS.savedir + '_output' + str(i)
		# simulate clean and noisy data
		(input_data_train, y_clean_train, y_noisy_train,
		y_clean_test_vec, y_noisy_test_vec, x_test_vec) = make_RNN_data2(
			sim_model, tspan_train, tspan_test, sim_model_params,
			noise_frac=0.05, output_dir=output_dir, drive_system=False,
			n_test_sets=FLAGS.n_tests,
			f_get_state_inits=get_lorenz_inits,
			continue_trajectory=FLAGS.continue_trajectory)

		###### do train/test split #######
		# n_train = int(np.floor(train_frac*len(y_clean)))
		# y_clean_train = y_clean[:n_train]
		# y_clean_test = y_clean[n_train:]
		# y_noisy_train = y_noisy[:n_train]
		# y_noisy_test = y_noisy[n_train:]
		# x_train = input_data[:, :n_train]
		# x_test = input_data[:, n_train:]
		# y_list = [y_clean_train, y_noisy_train, y_clean_test, y_noisy_test]

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
		# initialize testSynch vec with correct dimensions
		if FLAGS.continue_trajectory:
			y_clean_testSynch_vec_norm = np.copy(y_clean_test_vec[:,0,None])
			y_noisy_testSynch_vec_norm = np.copy(y_noisy_test_vec[:,0,None])
		else:
			y_clean_testSynch_vec_norm = np.copy(y_clean_test_vec[:,:ntsynch,:])
			y_noisy_testSynch_vec_norm = np.copy(y_noisy_test_vec[:,:ntsynch,:])
		for k in range(FLAGS.n_tests):
			y_clean_test_vec_norm[k,:,:] = f_normalize_minmax(normz_info, y_clean_test_vec[k,ntsynch:,:])
			y_noisy_test_vec_norm[k,:,:] = f_normalize_minmax(normz_info, y_noisy_test_vec[k,ntsynch:,:])
			if FLAGS.continue_trajectory:
				# in this case, just use the last training element as the synch element
				y_clean_testSynch_vec_norm[k,:,:] = y_clean_train_norm[-1,:]
				y_noisy_testSynch_vec_norm[k,:,:] = y_noisy_train_norm[-1,:]
			else:
				y_clean_testSynch_vec_norm[k,:,:] = f_normalize_minmax(normz_info, y_clean_test_vec[k,:ntsynch,:])
				y_noisy_testSynch_vec_norm[k,:,:] = f_normalize_minmax(normz_info, y_noisy_test_vec[k,:ntsynch,:])

		########## NOW start running RNN fits ############
		for n in range(FLAGS.n_experiments):
			for hidden_size in [50]:
				#### run vanilla RNN ####
				forward = forward_chaos_pureML
				# train on clean data
				run_output_dir = output_dir + '/iter{0}'.format(n) + '/vanillaRNN_clean_hs{0}'.format(hidden_size)
				all_dirs.append(run_output_dir)
				if not os.path.exists(run_output_dir+'/rnn_fit_ode_TEST_{0}.png'.format(FLAGS.n_tests-1)):
					# torch.manual_seed(0)
					train_chaosRNN(forward,
						y_clean_train_norm, y_clean_train_norm,
						y_clean_test_vec_norm, y_noisy_test_vec_norm,
						y_clean_testSynch_vec_norm, y_clean_testSynch_vec_norm,
						rnn_model_params, hidden_size, n_epochs, lr,
						run_output_dir, normz_info, rnn_sim_model,
						stack_hidden=False, stack_output=False,
						compute_kl=FLAGS.compute_kl)

			#### run mechRNN w/ BAD parameter ###
			forward = forward_chaos_hybrid_full

			for eps_badness in np.random.permutation([0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.4, 1]):
			# for eps_badness in [0, 0.01, 0.02]:
				rnn_BAD_model_params = {'state_names': ['x','y','z'], 'state_init':state_init, 'delta_t':delta_t, 'ode_params':(a, b*(1+eps_badness), c)}

				# train on clean data
				run_output_dir = output_dir + '/iter{0}'.format(n) + '/mechRNN_epsBadness{0}_clean_hs{1}'.format(eps_badness, hidden_size)
				all_dirs.append(run_output_dir)

				if not os.path.exists(run_output_dir+'/rnn_fit_ode_TEST_{0}.png'.format(FLAGS.n_tests-1)):
					# torch.manual_seed(0)
					train_chaosRNN(forward,
						y_clean_train_norm, y_clean_train_norm,
						y_clean_test_vec_norm, y_noisy_test_vec_norm,
						y_clean_testSynch_vec_norm, y_clean_testSynch_vec_norm,
						rnn_BAD_model_params, hidden_size, n_epochs, lr,
						run_output_dir, normz_info_clean, rnn_sim_model,
						compute_kl=FLAGS.compute_kl)

				# GP ONLY
				for gp_style in [1,2,3,4]:
					run_output_dir = output_dir + '/iter{0}'.format(n) + '/hybridGPR{2}_epsBadness{0}_clean_hs{1}'.format(eps_badness, hidden_size, gp_style)
					all_dirs.append(run_output_dir)
					if not os.path.exists(run_output_dir+'/fit_ode_TEST_{0}.png'.format(FLAGS.n_tests-1)):
						# torch.manual_seed(0)
						train_chaosRNN(forward,
							y_clean_train_norm, y_clean_train_norm,
							y_clean_test_vec_norm, y_noisy_test_vec_norm,
							y_clean_testSynch_vec_norm, y_clean_testSynch_vec_norm,
							rnn_BAD_model_params, hidden_size, n_epochs, lr,
							run_output_dir, normz_info_clean, rnn_sim_model,
							compute_kl=FLAGS.compute_kl, gp_only=True, gp_style=gp_style)

			# plot comparative training errors
			my_dirs = [d for d in all_dirs if "clean" in d]
			compare_fits(my_dirs, output_fname=output_dir+'/model_comparisons_clean')
			# extract_epsilon_performance(my_dirs, output_fname=output_dir+'/epsilon_comparison_clean')
			# compare_fits([d for d in all_dirs if "noisy" in d], output_fname=output_dir+'/model_comparisons_noisy')

if __name__ == '__main__':
	main()

