from utils import *
import numpy as np
import torch
import argparse
from odelibrary import L96M #(from file import class)

parser = argparse.ArgumentParser(description='mechRNN')
parser.add_argument('--epoch', type=int, default=4, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--delta_t', type=float, default=0.01, help='time step of simulation')
parser.add_argument('--t_train', type=float, default=4, help='length of train simulation')
parser.add_argument('--t_test', type=float, default=4, help='length of test simulation')
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
parser.add_argument('--ode_int_method', type=str, default='BDF', help='See scipy solve_ivp documentation for options.')
parser.add_argument('--ode_int_atol', type=float, default=1.5e-8, help='This is a much higher-fidelity tolerance than defaults for solve_ivp')
parser.add_argument('--ode_int_rtol', type=float, default=1.5e-8, help='This is a much higher-fidelity tolerance than defaults for solve_ivp')
parser.add_argument('--fix_seed', type=str2bool, default=False, help='Set to True if you want the call of this script to be reproducible (seed is set by last element of save_dir)')
parser.add_argument('--K', type=int, default=4, help='Dimension of Slow Variables')

parser.add_argument('--hidden_size', type=int, default=50, help='size of RNN hidden dimension')
parser.add_argument('--learn_residuals', type=str2bool, default=True, help='whether or not to learn a residual of a physical model')

FLAGS = parser.parse_args()

def main():
	output_dir = FLAGS.savedir

	########## NOW start running RNN fits ############
	#### run RNNs w/ SLOW system ###
	rnn_state_init = state_init[:l96m_AVAIL.K]
	rnn_sim_model = l96m_AVAIL.slow
	rnn_BAD_model_params = {'state_names': state_names, 'state_init':rnn_state_init, 'delta_t':delta_t, 'smaller_delta_t': min(delta_t, delta_t), 'ode_params':param_tuple, 'time_avg_norm':0.529, 'ode_int_method':FLAGS.ode_int_method, 'ode_int_rtol':FLAGS.ode_int_rtol, 'ode_int_atol':FLAGS.ode_int_atol, 'ode_int_max_step':np.inf}

	# Model Free GPR (map slow -> slow), residuals=False, gp_style=1
	rnn_model_params['learn_residuals_rnn'] = learn_residuals

	run_output_dir = output_dir + '/ModelFreeGPR_residual{1}_clean_hs{0}'.format(hidden_size, learn_residuals)
	all_dirs.append(run_output_dir)
	if not os.path.exists(run_output_dir+'/fit_ode_TEST_{0}.png'.format(FLAGS.n_tests-1)):
		# torch.manual_seed(0)
		train_chaosRNN(forward_chaos_pureML,
			y_clean_train_norm, y_noisy_train_norm,
			y_clean_test_vec_norm, y_noisy_test_vec_norm,
			y_clean_testSynch_vec_norm, y_noisy_testSynch_vec_norm,
			rnn_model_params, hidden_size, n_epochs, lr,
			run_output_dir, normz_info, rnn_sim_model,
			stack_hidden=False, stack_output=False,
			compute_kl=FLAGS.compute_kl, alpha_list=[FLAGS.alpha],
			gp_style=1, gp_resid=False, gp_only=True,
			plot_state_indices=plot_state_indices_SLOW)

	# GPR w/ residuals (learn_flow=False) gp_style 1, 2, and 3

	# GPR w/out residuals (learn_flow=False) gp_style 2 and 3
	for learn_flow in [False]:
		for learn_residuals in [True,False]:
			if learn_residuals:
				gp_list = [1,2,3] #GPR 1 is only a function of measured state, so it is model-free without residuals
			else:
				gp_list = [2,3]

			for gp_style in gp_list:
				run_output_dir = output_dir + '/hybridGPR{1}_residual{2}_learnflow{3}_clean_hs{0}'.format(hidden_size, gp_style, learn_residuals, learn_flow)
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

			if not learn_flow:
				# mechRNN
				run_output_dir = output_dir + '/mechRNN_residual{1}_learnflow{2}_clean_hs{0}'.format(hidden_size, learn_residuals, learn_flow)
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

				# vanillaRNN on bad-model's residuals
				run_output_dir = os.path.join(output_dir + '/vanillaRNN_residual{1}_clean_hs{0}'.format(hidden_size, learn_residuals)
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


if __name__ == '__main__':
	main()

