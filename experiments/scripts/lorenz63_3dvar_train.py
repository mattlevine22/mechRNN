from utils import *
# from utils import make_RNN_data, get_lorenz_inits, lorenz63
import numpy as np
import torch
import argparse
import pdb

parser = argparse.ArgumentParser(description='3DVAR')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--eps', type=float, default=1, help='observation noise coefficient')
parser.add_argument('--eta', type=float, default=0.1, help='observation noise coefficient')
parser.add_argument('--delta_t', type=float, default=0.01, help='time step of simulation')
parser.add_argument('--t_end', type=float, default=500, help='length of simulation')
parser.add_argument('--model_solver', default=lorenz63, help='ode function')
parser.add_argument('--output_dir', type=str, default='default_output', help='filename for generated data')
parser.add_argument('--training_data_filename', type=str, default='training_data', help='filename for generated data')
parser.add_argument('--testing_data_filename', type=str, default='training_data', help='filename for generated data')
parser.add_argument('--train_input_index', type=int, default=0, help='training trajectory index')
FLAGS = parser.parse_args()

def main():
	np.random.seed()

	# load a specific training trajectory + associated true/bad initial conditions
	npzfileTRAIN = np.load('{0}/{1}'.format(FLAGS.output_dir,FLAGS.training_data_filename))
	y_clean_TRAIN = npzfileTRAIN['y_clean'][FLAGS.train_input_index,:]
	y_noisy_TRAIN = npzfileTRAIN['y_noisy'][FLAGS.train_input_index,:]
	cheater_state_init_TRAIN = npzfileTRAIN['true_state_init'][FLAGS.train_input_index,:]
	random_state_init_TRAIN = npzfileTRAIN['random_state_init'][FLAGS.train_input_index,:]

	# load all testing trajectories + associated true/bad initial conditions
	npzfileTEST = np.load('{0}/{1}'.format(FLAGS.output_dir,FLAGS.testing_data_filename))
	y_clean_TEST = npzfileTEST['y_clean']
	y_noisy_TEST = npzfileTEST['y_noisy']
	cheater_state_init_TEST = npzfileTEST['true_state_init']
	random_state_init_TEST = npzfileTEST['random_state_init']

	# run settings
	n_avg = 10 # number of training iterations to average over to extract a useful K
	eps = FLAGS.eps
	eta = FLAGS.eta
	lr = FLAGS.lr # learning rate
	delta_t = FLAGS.delta_t #0.01
	tspan = np.arange(0,FLAGS.t_end,delta_t)  #np.arange(0,10000,delta_t)
	sim_model = FLAGS.model_solver

	H_obs = np.array([[1,0,0]])
	G_assim = H_obs.T/(1+eta)
	(a,b,c) = [10, 28, 8/3]

	assimilation_model_params = {'state_names': ['x','y','z'], 'state_init':random_state_init_TRAIN, 'delta_t':delta_t, 'smaller_delta_t': min(delta_t, delta_t), 'ode_params':(a, b, c), 'time_avg_norm':0.529, 'mxstep':0}

	# #### 3DVAR with perfect model

	# # Train
	# run_output_dir = '{0}/PerfectModel/Train{1}'.format(FLAGS.output_dir, FLAGS.train_input_index)
	# run_3DVAR(y_clean_TRAIN, y_noisy_TRAIN, H_obs, eta, G_assim, delta_t,
	# 	sim_model, assimilation_model_params, lr,
	# 	run_output_dir, learn_assim=False, inits=random_state_init_TRAIN, eps=eps)
	# # Test
	# for n_test in range(y_clean_TEST.shape[0]):
	# 	run_output_dir = '{0}/PerfectModel/Train{1}/Test{2}'.format(FLAGS.output_dir, FLAGS.train_input_index, n_test)
	# 	run_3DVAR(y_clean_TEST[n_test,:], y_noisy_TEST[n_test,:], H_obs, eta, G_assim, delta_t,
	# 		sim_model, assimilation_model_params, lr,
	# 		run_output_dir, learn_assim=False, inits=random_state_init_TEST[n_test], eps=eps)


	# #### 3D VAR with perfect model + learn the assimilation matrix

	# # Train
	# run_output_dir_TRAIN = '{0}/PerfectModel_learnAssimilation/Train{1}'.format(FLAGS.output_dir, FLAGS.train_input_index)
	# run_3DVAR(y_clean_TRAIN, y_noisy_TRAIN, H_obs, eta, G_assim, delta_t,
	# 	sim_model, assimilation_model_params, lr,
	# 	run_output_dir_TRAIN, learn_assim=True, inits=random_state_init_TRAIN, eps=eps)
	# # Test
	# npzfile = np.load(run_output_dir_TRAIN + '/output.npz')
	# G_assim_LEARNED = np.mean(npzfile['G_assim_history'][-n_avg:,:,None],axis=0)
	# for n_test in range(y_clean_TEST.shape[0]):
	# 	run_output_dir_TEST = '{0}/PerfectModel_learnAssimilation/Train{1}/Test{2}'.format(FLAGS.output_dir, FLAGS.train_input_index, n_test)
	# 	run_3DVAR(y_clean_TEST[n_test,:], y_noisy_TEST[n_test,:], H_obs, eta, G_assim_LEARNED, delta_t,
	# 		sim_model, assimilation_model_params, lr,
	# 		run_output_dir_TEST, learn_assim=False, inits=random_state_init_TEST[n_test], eps=eps)


	## BAD MODEL PARAMETERS NOW ##
	for eps_badness in np.arange(0,0.2,0.02):
		assimilation_model_params['ode_params'] = (a, b*(1+eps_badness), c)

		#### 3D VAR with eps-bad model + standard assimilation matrix
		# Train
		run_output_dir_TRAIN = '{0}/BadModel_eps{1}_standardAssimilation/Train{2}'.format(FLAGS.output_dir, eps_badness, FLAGS.train_input_index)
		run_3DVAR(y_clean_TRAIN, y_noisy_TRAIN, H_obs, eta, G_assim, delta_t,
			sim_model, assimilation_model_params, lr,
			run_output_dir_TRAIN, learn_assim=False, inits=random_state_init_TRAIN, eps=eps)
		# Test
		npzfile = np.load(run_output_dir_TRAIN + '/output.npz')
		G_assim_LEARNED = np.mean(npzfile['G_assim_history'][-n_avg:,:,None],axis=0)
		for n_test in range(y_clean_TEST.shape[0]):
			run_output_dir_TEST = '{0}/BadModel_eps{1}_standardAssimilation/Train{2}/Test{3}'.format(FLAGS.output_dir, eps_badness, FLAGS.train_input_index, n_test)
			run_3DVAR(y_clean_TEST[n_test,:], y_noisy_TEST[n_test,:], H_obs, eta, G_assim_LEARNED, delta_t,
				sim_model, assimilation_model_params, lr,
				run_output_dir_TEST, learn_assim=False, inits=random_state_init_TEST[n_test], eps=eps)

		#### 3D VAR with eps-bad model + learned assimilation matrix
		# Train
		run_output_dir_TRAIN = '{0}/BadModel_eps{1}_learnAssimilation/Train{2}'.format(FLAGS.output_dir, eps_badness, FLAGS.train_input_index)
		run_3DVAR(y_clean_TRAIN, y_noisy_TRAIN, H_obs, eta, G_assim, delta_t,
			sim_model, assimilation_model_params, lr,
			run_output_dir_TRAIN, learn_assim=True, inits=random_state_init_TRAIN, eps=eps)
		# Test
		npzfile = np.load(run_output_dir_TRAIN + '/output.npz')
		G_assim_LEARNED = np.mean(npzfile['G_assim_history'][-n_avg:,:,None],axis=0)
		for n_test in range(y_clean_TEST.shape[0]):
			run_output_dir_TEST = '{0}/BadModel_eps{1}_learnAssimilation/Train{2}/Test{3}'.format(FLAGS.output_dir, eps_badness, FLAGS.train_input_index, n_test)
			run_3DVAR(y_clean_TEST[n_test,:], y_noisy_TEST[n_test,:], H_obs, eta, G_assim_LEARNED, delta_t,
				sim_model, assimilation_model_params, lr,
				run_output_dir_TEST, learn_assim=False, inits=random_state_init_TEST[n_test], eps=eps)


if __name__ == '__main__':
	main()

