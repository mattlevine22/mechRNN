from utils import *
import numpy as np
import torch
import argparse
from L96M import L96M #(from file import class)


L96Mdefault = L96M()

parser = argparse.ArgumentParser(description='mechRNN')
parser.add_argument('--epoch', type=int, default=4, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--delta_t', type=float, default=0.01, help='time step of simulation')
parser.add_argument('--t_train', type=float, default=10, help='length of train simulation')
# parser.add_argument('--n_train_points', type=float, default=None, help='total number of train+testing data points. Default is to have this setting inactive (i.e None)')
parser.add_argument('--t_test', type=float, default=10, help='length of test simulation')
parser.add_argument('--t_test_synch', type=float, default=1, help='length of test simulation')
parser.add_argument('--savedir', type=str, default='default_output_eps', help='parent dir of output')
# parser.add_argument('--model_solver', default=lorenz63, help='ode function')
parser.add_argument('--drive_system', type=str2bool, default=False, help='whether to force the system with a time-dependent driver')
parser.add_argument('--n_tests', type=int, default=1, help='number of independent testing sets to use')
parser.add_argument('--n_experiments', type=int, default=1, help='number of sim/fitting experiments to do')
parser.add_argument('--random_state_inits', type=str2bool, default=True, help='whether to randomly initialize initial conditions for simulated trajectory')
parser.add_argument('--compute_kl', type=str2bool, default=False, help='whether to compute KL divergence between test set density and predicted density')
parser.add_argument('--continue_trajectory', type=str2bool, default=False, help='if true, ignore n_tests and synch length. Instead, simply continue train trajectory into the test trajectory.')
parser.add_argument('--noisy_training', type=str2bool, default=False, help='Optionally add measurement noise to training set.')
parser.add_argument('--noisy_testing', type=str2bool, default=False, help='Optionally add measurement noise to synch and test sets.')
parser.add_argument('--noise_frac', type=float, default=0.01, help='Relative SD of additive gaussian measurement noise')
parser.add_argument('--alpha', type=float, default=1e-10, help='Alpha parameter for Gaussian Process Regression (scales w/ measurement noise)')
parser.add_argument('--ode_int_method', type=str, default='RK45', help='See scipy solve_ivp documentation for options.')
parser.add_argument('--ode_int_atol', type=float, default=1.5e-6, help='This is a much higher-fidelity tolerance than defaults for solve_ivp')
parser.add_argument('--ode_int_rtol', type=float, default=1.5e-3, help='This is a much higher-fidelity tolerance than defaults for solve_ivp')
parser.add_argument('--ode_int_max_step', type=float, default=1.5e-3, help='This is a much higher-fidelity tolerance than defaults for solve_ivp')
parser.add_argument('--datagen_ode_int_method', type=str, default='Radau', help='See scipy solve_ivp documentation for options.')
parser.add_argument('--datagen_ode_int_atol', type=float, default=1.5e-6, help='This is a much higher-fidelity tolerance than defaults for solve_ivp')
parser.add_argument('--datagen_ode_int_rtol', type=float, default=1.5e-3, help='This is a much higher-fidelity tolerance than defaults for solve_ivp')
parser.add_argument('--datagen_ode_int_max_step', type=float, default=1.5e-3, help='This is a much higher-fidelity tolerance than defaults for solve_ivp')
parser.add_argument('--fix_seed', type=str2bool, default=False, help='Set to True if you want the call of this script to be reproducible (seed is set by last element of save_dir')
parser.add_argument('--run_RNN', type=str2bool, default=False, help='Set to True if you want to train RNNs')
parser.add_argument('--do_flow', type=str2bool, default=False, help='Set to True if you want to learn the vector field rhs of the ode')
parser.add_argument('--K', type=int, default=L96Mdefault.K, help='Dimension of Slow Variables')
parser.add_argument('--J', type=int, default=L96Mdefault.J, help='Dimension of Slow Variables')
parser.add_argument('--F', type=float, default=L96Mdefault.F, help='Forcing of L96 vars')
parser.add_argument('--hx', type=float, default=L96Mdefault.hx[0], help='coupling of slow vars to the fast vars')
parser.add_argument('--eps_power', type=int, default=-7, help='eps=2**(eps_power)')
parser.add_argument('--slow_only', type=str2bool, default=True, help='')

FLAGS = parser.parse_args()

def main():
	if FLAGS.fix_seed:
		np.random.seed(int(FLAGS.savedir[-1]))

	K = FLAGS.K
	J = FLAGS.J

	#python3 l96_sandbox.py --savedir ~/Downloads/l96_tests_v2/BDF_Init0 --delta_t 0.01 --t_train 10 --n_tests 1 --fix_seed True --t_test 10 --slow_only True
	# initialize TRUE model
	l96m_TRUE = L96M(K=K, J=J, F=FLAGS.F, hx=FLAGS.hx, eps = 2**(FLAGS.eps_power)) # model used to generate Train/Test data
	l96m_AVAIL = L96M(K=K, J=J, F=FLAGS.F, hx=FLAGS.hx, eps = 2**(FLAGS.eps_power)) # model used as a predictor

	plot_state_indices_FULL=[0, K]
	plot_state_indices_SLOW=[0, 1, K-2, K-1]

	get_l96_inits = l96m_TRUE.get_inits
	param_tuple = ()

	sim_model = l96m_TRUE.full

	# establish initial conditions
	my_state_inits = [np.squeeze(get_l96_inits(n=1))]

	# set up state names
	state_names = ['X_'+ str(k+1) for k in range(K)]
	for k in range(K):
	  state_names += ['Y_' + str(j+1) + ',' + str(k+1) for j in range(J)]


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
	drive_system = FLAGS.drive_system #False

	n_epochs = FLAGS.epoch #1

	# train_frac = FLAGS.train_frac #0.9995
	i = 0
	for state_init in my_state_inits:
		i += 1
		sim_model_params = {'state_names': state_names, 'state_init':state_init, 'delta_t':delta_t, 'smaller_delta_t': min(delta_t, delta_t), 'ode_params':param_tuple, 'time_avg_norm':0.529, 'ode_int_method':FLAGS.datagen_ode_int_method, 'ode_int_rtol':FLAGS.datagen_ode_int_rtol, 'ode_int_atol':FLAGS.datagen_ode_int_atol, 'ode_int_max_step':FLAGS.datagen_ode_int_max_step}
		# rnn_model_params = {'state_names': state_names, 'state_init':state_init, 'delta_t':delta_t, 'smaller_delta_t': min(delta_t, delta_t), 'ode_params':param_tuple, 'time_avg_norm':0.529, 'ode_int_method':FLAGS.ode_int_method, 'ode_int_rtol':FLAGS.ode_int_rtol, 'ode_int_atol':FLAGS.ode_int_atol, 'ode_int_max_step':np.inf}
		all_dirs = []

		# np.random.seed()
		if FLAGS.noisy_training:
			nm_train = 'noisy{0}'.format(FLAGS.noise_frac)
		else:
			nm_train = 'clean'

		if FLAGS.noisy_testing:
			nm_test = 'noisy{0}'.format(FLAGS.noise_frac)
		else:
			nm_test = 'clean'


		# master output directory name
		output_dir = FLAGS.savedir + '_output' + str(i) + '_{0}Train_{1}Test'.format(nm_train, nm_test)

		# write settings file
		main_dir = FLAGS.savedir[:FLAGS.savedir.rfind("/")]
		my_ind = FLAGS.savedir.rfind("/")
		if my_ind >= 0:
			main_dir = main_dir[:my_ind]
		try:
			os.mkdir(main_dir)
		except:
			# file exists
			pass

		settings_fname = main_dir + '/run_settings.txt'
		write_settings(FLAGS, settings_fname)

		# simulate clean and noisy data
		(input_data_train, y_clean_train, y_noisy_train,
		y_clean_test_vec, y_noisy_test_vec, x_test_vec) = make_RNN_data2(
			sim_model, tspan_train, tspan_test, sim_model_params,
			noise_frac=FLAGS.noise_frac, output_dir=output_dir, drive_system=False,
			n_test_sets=FLAGS.n_tests,
			f_get_state_inits=get_l96_inits,
			continue_trajectory=FLAGS.continue_trajectory,
			plot_state_indices=plot_state_indices_FULL)

		# only take slow variables in the simulation
		if FLAGS.slow_only:
			y_clean_train = y_clean_train[:,:K]
			y_noisy_train = y_noisy_train[:,:K]
			y_clean_test_vec = y_clean_test_vec[:,:,:K]
			y_noisy_test_vec = y_noisy_test_vec[:,:,:K]


		###### do train/test split #######
		# n_train = int(np.floor(train_frac*len(y_clean)))
		# y_clean_train = y_clean[:n_train]
		# y_clean_test = y_clean[n_train:]
		# y_noisy_train = y_noisy[:n_train]
		# y_noisy_test = y_noisy[n_train:]
		# x_train = input_data[:, :n_train]
		# x_test = input_data[:, n_train:]
		# y_list = [y_clean_train, y_noisy_train, y_clean_test, y_noisy_test]

		if not FLAGS.noisy_training:
			y_noisy_train = y_clean_train
		if not FLAGS.noisy_testing:
			# makes synch and test sets clean
			y_noisy_test_vec = y_clean_test_vec

		####### collect normalization information from TRAINING SET ONLY ######
		normz_info = {}
		normz_info['Ymax'] = np.max(y_noisy_train,axis=0)
		normz_info['Ymin'] = np.min(y_noisy_train,axis=0)
		normz_info['Ymean'] = np.mean(y_noisy_train)
		normz_info['Ysd'] = np.std(y_noisy_train)
		# normz_info['Xmean'] = np.mean(x_train)
		# normz_info['Xsd'] = np.std(x_train)

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
		#### run RNNs w/ SLOW system ###
		for n in range(FLAGS.n_experiments):
			for hidden_size in [50, 25, 10, 15, 100]:
				param_tuple = ()
				if FLAGS.slow_only:
					rnn_state_init = state_init[:K]
					rnn_sim_model = l96m_AVAIL.slow
				else:
					rnn_state_init = state_init
					rnn_sim_model = l96m_AVAIL.full

				rnn_BAD_model_params = {'state_names': state_names, 'state_init':rnn_state_init, 'delta_t':delta_t, 'smaller_delta_t': min(delta_t, delta_t), 'ode_params':param_tuple, 'time_avg_norm':0.529, 'ode_int_method':FLAGS.ode_int_method, 'ode_int_rtol':FLAGS.ode_int_rtol, 'ode_int_atol':FLAGS.ode_int_atol, 'ode_int_max_step':FLAGS.ode_int_max_step}


				# FIRST run bad ODE model alone
				run_output_dir = output_dir + '/iter{0}'.format(n) + '/pureODE_clean'
				all_dirs.append(run_output_dir)
				if not os.path.exists(run_output_dir+'/fit_ode_TEST_{0}.png'.format(FLAGS.n_tests-1)):
					# torch.manual_seed(0)
					train_chaosRNN(forward_chaos_pureML,
						y_clean_train_norm, y_noisy_train_norm,
						y_clean_test_vec_norm, y_noisy_test_vec_norm,
						y_clean_testSynch_vec_norm, y_noisy_testSynch_vec_norm,
						rnn_BAD_model_params, hidden_size, n_epochs, lr,
						run_output_dir, normz_info, rnn_sim_model,
						compute_kl=FLAGS.compute_kl,
						ode_only=True)


				# Model Free GPR (map slow -> slow), residuals=False, gp_style=1
				learn_residuals = False
				rnn_BAD_model_params['learn_residuals_rnn'] = learn_residuals

				run_output_dir = output_dir + '/iter{0}'.format(n) + '/ModelFreeGPR_residual{0}_clean'.format(learn_residuals)
				all_dirs.append(run_output_dir)
				if not os.path.exists(run_output_dir+'/fit_ode_TEST_{0}.png'.format(FLAGS.n_tests-1)):
					# torch.manual_seed(0)
					train_chaosRNN(forward_chaos_pureML,
						y_clean_train_norm, y_noisy_train_norm,
						y_clean_test_vec_norm, y_noisy_test_vec_norm,
						y_clean_testSynch_vec_norm, y_noisy_testSynch_vec_norm,
						rnn_BAD_model_params, hidden_size, n_epochs, lr,
						run_output_dir, normz_info, rnn_sim_model,
						stack_hidden=False, stack_output=False,
						compute_kl=FLAGS.compute_kl, alpha_list=[FLAGS.alpha],
						gp_style=1, gp_resid=False, gp_only=True,
						plot_state_indices=plot_state_indices_SLOW)

				# GPR w/ residuals (learn_flow=False) gp_style 1, 2, and 3

				# GPR w/out residuals (learn_flow=False) gp_style 2 and 3
				for learn_flow in [False]:
					for learn_residuals in [True,False]:
						rnn_BAD_model_params['learn_residuals_rnn'] = learn_residuals

						if learn_residuals:
							gp_list = [1,2,3] #GPR 1 is only a function of measured state, so it is model-free without residuals
						else:
							gp_list = []

						for gp_style in gp_list:
							run_output_dir = output_dir + '/iter{0}'.format(n) + '/hybridGPR{0}_residual{1}_learnflow{2}_clean'.format(gp_style, learn_residuals, learn_flow)
							all_dirs.append(run_output_dir)
							if not os.path.exists(run_output_dir+'/fit_ode_TEST_{0}.png'.format(FLAGS.n_tests-1)):
								# torch.manual_seed(0)
								train_chaosRNN(forward_chaos_pureML,
									y_clean_train_norm, y_noisy_train_norm,
									y_clean_test_vec_norm, y_noisy_test_vec_norm,
									y_clean_testSynch_vec_norm, y_noisy_testSynch_vec_norm,
									rnn_BAD_model_params, hidden_size, n_epochs, lr,
									run_output_dir, normz_info, rnn_sim_model,
									compute_kl=FLAGS.compute_kl, gp_only=True, gp_style=gp_style, alpha_list=[FLAGS.alpha],
									gp_resid=learn_residuals,
									learn_flow=learn_flow,
									plot_state_indices=plot_state_indices_SLOW)

						if FLAGS.run_RNN:
							if not learn_flow:
								# vanillaRNN on bad-model's residuals
								run_output_dir = output_dir + '/iter{0}'.format(n) + '/vanillaRNN_residual{1}_clean_hs{0}'.format(hidden_size, learn_residuals)
								all_dirs.append(run_output_dir)
								if not os.path.exists(run_output_dir+'/rnn_fit_ode_TEST_{0}.png'.format(FLAGS.n_tests-1)):
									# torch.manual_seed(0)
									train_chaosRNN(forward_chaos_pureML,
										y_clean_train_norm, y_noisy_train_norm,
										y_clean_test_vec_norm, y_noisy_test_vec_norm,
										y_clean_testSynch_vec_norm, y_noisy_testSynch_vec_norm,
										rnn_BAD_model_params, hidden_size, n_epochs, lr,
										run_output_dir, normz_info, rnn_sim_model,
										stack_hidden=False, stack_output=False,
										compute_kl=FLAGS.compute_kl, alpha_list=[FLAGS.alpha],
										plot_state_indices=plot_state_indices_SLOW)

							if not learn_flow and not learn_residuals:
								# mechRNN
								run_output_dir = output_dir + '/iter{0}'.format(n) + '/mechRNN_residual{1}_learnflow{2}_clean_hs{0}'.format(hidden_size, learn_residuals, learn_flow)
								all_dirs.append(run_output_dir)

								if not os.path.exists(run_output_dir+'/rnn_fit_ode_TEST_{0}.png'.format(FLAGS.n_tests-1)):
									# torch.manual_seed(0)
									train_chaosRNN(forward_chaos_hybrid_full,
										y_clean_train_norm, y_noisy_train_norm,
										y_clean_test_vec_norm, y_noisy_test_vec_norm,
										y_clean_testSynch_vec_norm, y_noisy_testSynch_vec_norm,
										rnn_BAD_model_params, hidden_size, n_epochs, lr,
										run_output_dir, normz_info, rnn_sim_model,
										compute_kl=FLAGS.compute_kl, alpha_list=[FLAGS.alpha],
										plot_state_indices=plot_state_indices_SLOW)


			# plot comparative training errors
			my_dirs = [d for d in all_dirs if "clean" in d]
			compare_fits(my_dirs, output_fname=output_dir+'/model_comparisons_clean')
			# extract_epsilon_performance(my_dirs, output_fname=output_dir+'/epsilon_comparison_clean')
			# compare_fits([d for d in all_dirs if "noisy" in d], output_fname=output_dir+'/model_comparisons_noisy')

if __name__ == '__main__':
	main()

