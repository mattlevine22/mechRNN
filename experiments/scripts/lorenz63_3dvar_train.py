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
parser.add_argument('--cheat', type=str2bool, default=True, help='cheating means using unobserved data (aka applying H_obs_hifi). If False, H_obs_hifi is inactive.')
FLAGS = parser.parse_args()


if not FLAGS.cheat:
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

	G_assim_magical = np.array([[0.08603118, 0.12466607, 0.00351079]]).T
	# I GET G_assim_magical with the following code:
	# G_all = np.zeros((10,3))
	# for n in range(10):
	# 	f = '/groups/astuart/mlevine/Sep11/BadModel_eps0.0_learnAssimilation/Train{0}/output.npz'.format(n)
	# 	npzfile = np.load(f)
	# 	G_all[n,:] = npzfile['G_assim_history_running_mean'][-1]
	# np.mean(G_all, axis=0)



	# #### 3DVAR with perfect model

	# # Train
	# run_output_dir = '{0}/PerfectModel/Train{1}'.format(FLAGS.output_dir, FLAGS.train_input_index)
	# run_3DVAR(y_clean_TRAIN, y_noisy_TRAIN, eta, G_assim_standard, delta_t,
	# 	sim_model, assimilation_model_params, lr,
	# 	run_output_dir,
			# H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
	# 	learn_assim=False, inits=random_state_init_TRAIN, eps=eps, cheat=FLAGS.cheat)
	# # Test
	# for n_test in range(y_clean_TEST.shape[0]):
	# 	run_output_dir = '{0}/PerfectModel/Train{1}/Test{2}'.format(FLAGS.output_dir, FLAGS.train_input_index, n_test)
	# 	run_3DVAR(y_clean_TEST[n_test,:], y_noisy_TEST[n_test,:], eta, G_assim_standard, delta_t,
	# 		sim_model, assimilation_model_params, lr,
			# H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
	# 		run_output_dir, learn_assim=False, inits=random_state_init_TEST[n_test], eps=eps, cheat=FLAGS.cheat)


	# #### 3D VAR with perfect model + learn the assimilation matrix

	# # Train
	# run_output_dir_TRAIN = '{0}/PerfectModel_learnAssimilation/Train{1}'.format(FLAGS.output_dir, FLAGS.train_input_index)
	# run_3DVAR(y_clean_TRAIN, y_noisy_TRAIN, eta, G_assim_init, delta_t,
	# 	sim_model, assimilation_model_params, lr,
	# 	run_output_dir_TRAIN,
	# 				H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
	# learn_assim=True, inits=random_state_init_TRAIN, eps=eps, cheat=FLAGS.cheat)
	# # Test
	# npzfile = np.load(run_output_dir_TRAIN + '/output.npz')
	# G_assim_LEARNED = np.mean(npzfile['G_assim_history'][-n_avg:,:,None],axis=0)
	# for n_test in range(y_clean_TEST.shape[0]):
	# 	run_output_dir_TEST = '{0}/PerfectModel_learnAssimilation/Train{1}/Test{2}'.format(FLAGS.output_dir, FLAGS.train_input_index, n_test)
	# 	run_3DVAR(y_clean_TEST[n_test,:], y_noisy_TEST[n_test,:], eta, G_assim_LEARNED, delta_t,
	# 		sim_model, assimilation_model_params, lr,
	# 		run_output_dir_TEST,
			# H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
	# 		learn_assim=False, inits=random_state_init_TEST[n_test], eps=eps, cheat=FLAGS.cheat)


	## BAD MODEL PARAMETERS NOW ##
	for eps_badness in [0.0]:#np.arange(0,0.04,0.02):
		assimilation_model_params['ode_params'] = (a, b*(1+eps_badness), c)

		#### 3D VAR with eps-bad model + standard assimilation matrix
		# Train
		run_output_dir_TRAIN = '{0}/BadModel_eps{1}_standardAssimilation/Train{2}'.format(FLAGS.output_dir, eps_badness, FLAGS.train_input_index)
		if not os.path.exists(run_output_dir_TRAIN):
			run_3DVAR(y_clean_TRAIN, y_noisy_TRAIN, eta, G_assim_standard, delta_t,
				sim_model, assimilation_model_params, lr,
				run_output_dir_TRAIN,
				H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
				learn_assim=False, inits=random_state_init_TRAIN, eps=eps, cheat=FLAGS.cheat)
			# Test
			npzfile = np.load(run_output_dir_TRAIN + '/output.npz')
			# G_assim_LEARNED = npzfile['G_assim_history_running_mean'][-1,:,None]
			for n_test in range(y_clean_TEST.shape[0]):
				run_output_dir_TEST = '{0}/BadModel_eps{1}_standardAssimilation/Train{2}/Test{3}'.format(FLAGS.output_dir, eps_badness, FLAGS.train_input_index, n_test)
				run_3DVAR(y_clean_TEST[n_test,:], y_noisy_TEST[n_test,:], eta, G_assim_standard, delta_t,
					sim_model, assimilation_model_params, lr,
					run_output_dir_TEST,
					H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
					learn_assim=False, inits=random_state_init_TEST[n_test], eps=eps, cheat=FLAGS.cheat)

		#### 3D VAR with eps-bad model + magical assimilation matrix
		# Train
		run_output_dir_TRAIN = '{0}/BadModel_eps{1}_magicalAssimilation/Train{2}'.format(FLAGS.output_dir, eps_badness, FLAGS.train_input_index)
		if not os.path.exists(run_output_dir_TRAIN):
			run_3DVAR(y_clean_TRAIN, y_noisy_TRAIN, eta, G_assim_magical, delta_t,
				sim_model, assimilation_model_params, lr,
				run_output_dir_TRAIN,
				H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
				learn_assim=False, inits=random_state_init_TRAIN, eps=eps, cheat=FLAGS.cheat)
			# Test
			npzfile = np.load(run_output_dir_TRAIN + '/output.npz')
			# G_assim_LEARNED = npzfile['G_assim_history_running_mean'][-1,:,None]
			for n_test in range(y_clean_TEST.shape[0]):
				run_output_dir_TEST = '{0}/BadModel_eps{1}_magicalAssimilation/Train{2}/Test{3}'.format(FLAGS.output_dir, eps_badness, FLAGS.train_input_index, n_test)
				run_3DVAR(y_clean_TEST[n_test,:], y_noisy_TEST[n_test,:], eta, G_assim_magical, delta_t,
					sim_model, assimilation_model_params, lr,
					run_output_dir_TEST,
					H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
					learn_assim=False, inits=random_state_init_TEST[n_test], eps=eps, cheat=FLAGS.cheat)


		#### 3D VAR with eps-bad model + learned assimilation matrix
		# h_list = [1e-6, 1e-4, 1e-2]
		# lr_G_list = [1e-3, 1e-2, 5e-2]
		h_list = [1e-6]
		lr_G_list = [1e-3]

		np.random.shuffle(h_list)
		np.random.shuffle(lr_G_list)

		for N_q_tries in [1,100]:
			for h in h_list:
				for lr_G in lr_G_list:
					# Train
					run_output_dir_TRAIN = '{0}/BadModel_eps{1}_learnAssimilationRANDOMinit+h{3}+lrG{4}+NQtries{5}/Train{2}'.format(FLAGS.output_dir, eps_badness, FLAGS.train_input_index, h, lr_G, N_q_tries)
					if not os.path.exists(run_output_dir_TRAIN):
						run_3DVAR(y_clean_TRAIN, y_noisy_TRAIN, eta, G_assim_init, delta_t,
							sim_model, assimilation_model_params, lr,
							run_output_dir_TRAIN,
							H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
							learn_assim=True, inits=random_state_init_TRAIN, eps=eps, cheat=FLAGS.cheat,
							h=h, lr_G=lr_G, N_q_tries=N_q_tries)
						# Test
						npzfile = np.load(run_output_dir_TRAIN + '/output.npz')
						G_assim_LEARNED = npzfile['G_assim_history_running_mean'][-1,:,None]
						for n_test in range(y_clean_TEST.shape[0]):
							run_output_dir_TEST = '{0}/BadModel_eps{1}_learnAssimilationRANDOMinit+h{4}+lrG{5}+NQtries{6}/Train{2}/Test{3}'.format(FLAGS.output_dir, eps_badness, FLAGS.train_input_index, n_test, h, lr_G, N_q_tries)
							run_3DVAR(y_clean_TEST[n_test,:], y_noisy_TEST[n_test,:], eta, G_assim_LEARNED, delta_t,
								sim_model, assimilation_model_params, lr,
								run_output_dir_TEST,
								H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
								learn_assim=False, inits=random_state_init_TEST[n_test], eps=eps, cheat=FLAGS.cheat,
								h=h, lr_G=lr_G, N_q_tries=N_q_tries)

					## INITIALIZE AT STANDARD G AND DO TRAIN/TEST
					# Train
					run_output_dir_TRAIN = '{0}/BadModel_eps{1}_learnAssimilationSTANDARDinit+h{3}+lrG{4}+NQtries{5}/Train{2}'.format(FLAGS.output_dir, eps_badness, FLAGS.train_input_index, h, lr_G, N_q_tries)
					if not os.path.exists(run_output_dir_TRAIN):
						run_3DVAR(y_clean_TRAIN, y_noisy_TRAIN, eta, G_assim_standard, delta_t,
							sim_model, assimilation_model_params, lr,
							run_output_dir_TRAIN,
							H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
							learn_assim=True, inits=random_state_init_TRAIN, eps=eps, cheat=FLAGS.cheat,
							h=h, lr_G=lr_G, N_q_tries=N_q_tries)
						# Test
						npzfile = np.load(run_output_dir_TRAIN + '/output.npz')
						G_assim_LEARNED = npzfile['G_assim_history_running_mean'][-1,:,None]
						for n_test in range(y_clean_TEST.shape[0]):
							run_output_dir_TEST = '{0}/BadModel_eps{1}_learnAssimilationSTANDARDinit+h{4}+lrG{5}+NQtries{6}/Train{2}/Test{3}'.format(FLAGS.output_dir, eps_badness, FLAGS.train_input_index, n_test, h, lr_G, N_q_tries)
							run_3DVAR(y_clean_TEST[n_test,:], y_noisy_TEST[n_test,:], eta, G_assim_LEARNED, delta_t,
								sim_model, assimilation_model_params, lr,
								run_output_dir_TEST,
								H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
								learn_assim=False, inits=random_state_init_TEST[n_test], eps=eps, cheat=FLAGS.cheat,
								h=h, lr_G=lr_G, N_q_tries=N_q_tries)

					## INITIALIZE AT KNOWN MAGICAL G AND DO TRAIN/TEST
					# Train
					run_output_dir_TRAIN = '{0}/BadModel_eps{1}_learnAssimilationMAGICALinit+h{3}+lrG{4}+NQtries{5}/Train{2}'.format(FLAGS.output_dir, eps_badness, FLAGS.train_input_index, h, lr_G, N_q_tries)
					if not os.path.exists(run_output_dir_TRAIN):
						run_3DVAR(y_clean_TRAIN, y_noisy_TRAIN, eta, G_assim_magical, delta_t,
							sim_model, assimilation_model_params, lr,
							run_output_dir_TRAIN,
							H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
							learn_assim=True, inits=random_state_init_TRAIN, eps=eps, cheat=FLAGS.cheat,
							h=h, lr_G=lr_G, N_q_tries=N_q_tries)
						# Test
						npzfile = np.load(run_output_dir_TRAIN + '/output.npz')
						G_assim_LEARNED = npzfile['G_assim_history_running_mean'][-1,:,None]
						for n_test in range(y_clean_TEST.shape[0]):
							run_output_dir_TEST = '{0}/BadModel_eps{1}_learnAssimilationMAGICALinit+h{4}+lrG{5}+NQtries{6}/Train{2}/Test{3}'.format(FLAGS.output_dir, eps_badness, FLAGS.train_input_index, n_test, h, lr_G, N_q_tries)
							run_3DVAR(y_clean_TEST[n_test,:], y_noisy_TEST[n_test,:], eta, G_assim_LEARNED, delta_t,
								sim_model, assimilation_model_params, lr,
								run_output_dir_TEST,
								H_obs_lowfi=H_obs_lowfi, H_obs_hifi=H_obs_hifi, noisy_hifi=FLAGS.noisy_hifi,
								learn_assim=False, inits=random_state_init_TEST[n_test], eps=eps, cheat=FLAGS.cheat,
								h=h, lr_G=lr_G, N_q_tries=N_q_tries)


if __name__ == '__main__':
	main()

