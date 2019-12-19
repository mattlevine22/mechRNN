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
parser.add_argument('--H_obs_lowfi', type=str2array, default=np.array([[1,0,0]]), help='low-fidelity observation operator')
parser.add_argument('--H_obs_hifi', type=str2array, default=np.array([[1,0,0],[0,1,0],[0,0,1]]), help='hi-fidelity observation operator')
parser.add_argument('--G_init_sd', type=float, default=0.1, help='standard deviation for random initialization of assimilation matrix')
parser.add_argument('--noisy_hifi', type=str2bool, default=False, help='When cheating, noisy_hifi optionally adds measurement noise to the hi-fi observations provided by H_obs_hifi.')
parser.add_argument('--optim_type', type=str, default='NelderMead', help='NA')
parser.add_argument('--optim_full_state', type=str2bool, default=False, help='NA')
parser.add_argument('--n_nelder_inits', type=int, default=1, help='NA')
parser.add_argument('--max_nelder_sols', type=int, default=None, help='Number of function calls allowed for nelder mead')
# parser.add_argument('--cheat', type=str2bool, default=False, help='cheating means using unobserved data (aka applying H_obs_hifi). If False, H_obs_hifi is inactive.')
# parser.add_argument('--new_cheat', type=str2bool, default=False, help='cheating means using unobserved data (aka applying H_obs_hifi). If False, H_obs_hifi is inactive.')
FLAGS = parser.parse_args()

# if FLAGS.optim_type is not None:
FLAGS.cheat = False
FLAGS.new_cheat = False
learn_assim = True
full_sequence = True
optim_full_state = FLAGS.optim_full_state
# if FLAGS.optim_type=='GradientDescent':
# 	full_sequence = False
# 	optim_full_state = True
# 	FLAGS.cheat = True
# elif FLAGS.optim_type=='NelderMead':

if optim_full_state:
	state_str = 'FullState'
else:
	state_str = 'PartialState'


if not FLAGS.cheat and not FLAGS.new_cheat:
	# if not cheating, then hifi and lowfi observations must be the same
	FLAGS.H_obs_hifi = FLAGS.H_obs_lowfi

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

	G_assim_standard = np.array([[1,0,0]]).T/(1+eta)
	H_obs_lowfi = FLAGS.H_obs_lowfi
	H_obs_hifi = FLAGS.H_obs_hifi
	G_assim_init_fname = '{0}/G_assim_init_Train{1}.npy'.format(FLAGS.output_dir, FLAGS.train_input_index)
	if os.path.exists(G_assim_init_fname):
		G_assim_init = np.load(G_assim_init_fname)
	else:
		G_assim_init = np.random.multivariate_normal(mean=[0,0,0], cov=(FLAGS.G_init_sd**2)*np.eye(3)).T[:,None]
		## Store G_assim_init in main directory so that it is the same random initial conditions across all runs of the same TrainX:
		np.save(G_assim_init_fname, G_assim_init)

	(a,b,c) = [10, 28, 8/3]

	assimilation_model_params = {'state_names': ['x','y','z'], 'state_init':random_state_init_TRAIN, 'delta_t':delta_t, 'smaller_delta_t': min(delta_t, delta_t), 'ode_params':(a, b, c), 'time_avg_norm':0.529, 'mxstep':0}

	eps_badness = 0

	# Gradient Descent MEAN Init
	run_output_dir_TRAIN = '{0}/BadModel_eps{1}_{2}_{3}_GradientDescentMEANInit/Train{4}'.format(FLAGS.output_dir, FLAGS.optim_type, eps_badness, state_str, FLAGS.train_input_index)
	if not os.path.exists(run_output_dir_TRAIN):
		G_assim_init = np.load('{0}/BadModel_eps{1}_{2}_{3}_RandomInit/Kmean.npy'.format(FLAGS.output_dir, 'GradientDescent', eps_badness, 'FullState'))
		# G_assim_init = np.array([[1,0,0]]).T/(1+my_eta)

		# G_assim_init = np.random.multivariate_normal(mean=[0,0,0], cov=(0.1*FLAGS.G_init_sd**2)*np.eye(3)).T[:,None]
		print('Init=',G_assim_init)
		G_assim_LEARNED = run_3DVAR(y_clean_TRAIN, y_noisy_TRAIN, eta, G_assim_init, delta_t,
			sim_model, assimilation_model_params, lr,
			run_output_dir_TRAIN,
			H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
			learn_assim=True, inits=random_state_init_TRAIN, eps=eps, cheat=FLAGS.cheat,
			full_sequence=full_sequence, optimization=FLAGS.optim_type, optim_full_state=optim_full_state,
			random_nelder_inits=False, max_nelder_sols=FLAGS.max_nelder_sols)
		# Test
		# npzfile = np.load(run_output_dir_TRAIN + '/output.npz')
		# G_assim_LEARNED = npzfile['G_assim_history_running_mean'][-1,:,None]
		G_assim_LEARNED = G_assim_LEARNED[:,None]
		for n_test in range(y_clean_TEST.shape[0]):
			run_output_dir_TEST = '{0}/BadModel_eps{1}_{2}_{3}_GradientDescentMEANInit/Train{4}/Test{5}'.format(FLAGS.output_dir, FLAGS.optim_type, eps_badness, state_str, FLAGS.train_input_index, n_test)
			run_3DVAR(y_clean_TEST[n_test,:], y_noisy_TEST[n_test,:], eta, G_assim_LEARNED, delta_t,
				sim_model, assimilation_model_params, lr,
				run_output_dir_TEST,
				H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
				learn_assim=False, inits=random_state_init_TEST[n_test], eps=eps, cheat=FLAGS.cheat)

	# Gradient Descent run-specific Init
	run_output_dir_TRAIN = '{0}/BadModel_eps{1}_{2}_{3}_GradientDescentInit/Train{4}'.format(FLAGS.output_dir, FLAGS.optim_type, eps_badness, state_str, FLAGS.train_input_index)
	if not os.path.exists(run_output_dir_TRAIN):
		fname = '{0}/BadModel_eps{1}_{2}_{3}_RandomInit/Train{4}/output.npz'.format(FLAGS.output_dir, 'GradientDescent', eps_badness, 'FullState', FLAGS.train_input_index)
		foo = np.load(fname)
		G_assim_init = foo['G_assim_history_running_mean'][-1]
		# G_assim_init = np.array([[1,0,0]]).T/(1+my_eta)
		# G_assim_init = np.random.multivariate_normal(mean=[0,0,0], cov=(0.1*FLAGS.G_init_sd**2)*np.eye(3)).T[:,None]
		print('Init=',G_assim_init)
		G_assim_LEARNED = run_3DVAR(y_clean_TRAIN, y_noisy_TRAIN, eta, G_assim_init, delta_t,
			sim_model, assimilation_model_params, lr,
			run_output_dir_TRAIN,
			H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
			learn_assim=True, inits=random_state_init_TRAIN, eps=eps, cheat=FLAGS.cheat,
			full_sequence=full_sequence, optimization=FLAGS.optim_type, optim_full_state=optim_full_state,
			random_nelder_inits=False, max_nelder_sols=FLAGS.max_nelder_sols)
		# Test
		# npzfile = np.load(run_output_dir_TRAIN + '/output.npz')
		# G_assim_LEARNED = npzfile['G_assim_history_running_mean'][-1,:,None]
		G_assim_LEARNED = G_assim_LEARNED[:,None]
		for n_test in range(y_clean_TEST.shape[0]):
			run_output_dir_TEST = '{0}/BadModel_eps{1}_{2}_{3}_GradientDescentInit/Train{4}/Test{5}'.format(FLAGS.output_dir, FLAGS.optim_type, eps_badness, state_str, FLAGS.train_input_index, n_test)
			run_3DVAR(y_clean_TEST[n_test,:], y_noisy_TEST[n_test,:], eta, G_assim_LEARNED, delta_t,
				sim_model, assimilation_model_params, lr,
				run_output_dir_TEST,
				H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
				learn_assim=False, inits=random_state_init_TEST[n_test], eps=eps, cheat=FLAGS.cheat)



	# Random Inits
	run_output_dir_TRAIN = '{0}/BadModel_eps{1}_{2}_{3}_RandomInit/Train{4}'.format(FLAGS.output_dir, FLAGS.optim_type, eps_badness, state_str, FLAGS.train_input_index)
	if not os.path.exists(run_output_dir_TRAIN):
		G_assim_init = np.random.multivariate_normal(mean=[0,0,0], cov=(0.1*FLAGS.G_init_sd**2)*np.eye(3)).T[:,None]
		print('Init=',G_assim_init)
		G_assim_LEARNED = run_3DVAR(y_clean_TRAIN, y_noisy_TRAIN, eta, G_assim_init, delta_t,
			sim_model, assimilation_model_params, lr,
			run_output_dir_TRAIN,
			H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
			learn_assim=True, inits=random_state_init_TRAIN, eps=eps, cheat=FLAGS.cheat,
			full_sequence=full_sequence, optimization=FLAGS.optim_type, optim_full_state=optim_full_state,
			random_nelder_inits=True, n_nelder_inits=FLAGS.n_nelder_inits, max_nelder_sols=FLAGS.max_nelder_sols)
		# Test
		# npzfile = np.load(run_output_dir_TRAIN + '/output.npz')
		# G_assim_LEARNED = npzfile['G_assim_history_running_mean'][-1,:,None]
		G_assim_LEARNED = G_assim_LEARNED[:,None]
		for n_test in range(y_clean_TEST.shape[0]):
			run_output_dir_TEST = '{0}/BadModel_eps{1}_{2}_{3}_RandomInit/Train{4}/Test{5}'.format(FLAGS.output_dir, FLAGS.optim_type, eps_badness, state_str, FLAGS.train_input_index, n_test)
			run_3DVAR(y_clean_TEST[n_test,:], y_noisy_TEST[n_test,:], eta, G_assim_LEARNED, delta_t,
				sim_model, assimilation_model_params, lr,
				run_output_dir_TEST,
				H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
				learn_assim=False, inits=random_state_init_TEST[n_test], eps=eps, cheat=FLAGS.cheat)

	# Theory Inits
	for my_eta in [0.1, 0.5, 0.9]:
		run_output_dir_TRAIN = '{0}/BadModel_eps{1}_{2}_{3}_TheoryInit{4:.2f}/Train{5}'.format(FLAGS.output_dir, FLAGS.optim_type, eps_badness, state_str, my_eta, FLAGS.train_input_index)
		if not os.path.exists(run_output_dir_TRAIN):
			G_assim_init = np.array([[1,0,0]]).T/(1+my_eta)
			# G_assim_init = np.random.multivariate_normal(mean=[0,0,0], cov=(0.1*FLAGS.G_init_sd**2)*np.eye(3)).T[:,None]
			print('Init=',G_assim_init)
			G_assim_LEARNED = run_3DVAR(y_clean_TRAIN, y_noisy_TRAIN, my_eta, G_assim_init, delta_t,
				sim_model, assimilation_model_params, lr,
				run_output_dir_TRAIN,
				H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
				learn_assim=True, inits=random_state_init_TRAIN, eps=eps, cheat=FLAGS.cheat,
				full_sequence=full_sequence, optimization=FLAGS.optim_type, optim_full_state=optim_full_state,
				random_nelder_inits=False, max_nelder_sols=FLAGS.max_nelder_sols)
			# Test
			# npzfile = np.load(run_output_dir_TRAIN + '/output.npz')
			# G_assim_LEARNED = npzfile['G_assim_history_running_mean'][-1,:,None]
			G_assim_LEARNED = G_assim_LEARNED[:,None]
			for n_test in range(y_clean_TEST.shape[0]):
				run_output_dir_TEST = '{0}/BadModel_eps{1}_{2}_{3}_TheoryInit{4:.2f}/Train{5}/Test{6}'.format(FLAGS.output_dir, FLAGS.optim_type, eps_badness, state_str, my_eta, FLAGS.train_input_index, n_test)
				run_3DVAR(y_clean_TEST[n_test,:], y_noisy_TEST[n_test,:], my_eta, G_assim_LEARNED, delta_t,
					sim_model, assimilation_model_params, lr,
					run_output_dir_TEST,
					H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
					learn_assim=False, inits=random_state_init_TEST[n_test], eps=eps, cheat=FLAGS.cheat)


if __name__ == '__main__':
	main()

