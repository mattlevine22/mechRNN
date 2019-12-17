from utils import *
import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser(description='mechRNN')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--delta_t', type=float, default=0.01, help='time step of simulation')
parser.add_argument('--t_end', type=float, default=500, help='length of simulation')
parser.add_argument('--savedir', type=str, default='default_output', help='parent dir of output')
parser.add_argument('--model_solver', default=lorenz63, help='ode function')
FLAGS = parser.parse_args()

def main():
	eps = 1
	eta = 0.1
	H_obs = np.array([[1,0,0]])
	G_assim = H_obs.T/(1+eta)
	# P_assim = np.array([[1,0,0], [0,0,0], [0,0,0]])
	# Q_assim = np.array([[0,0,0], [0,1,0], [0,0,1]])

	(a,b,c) = [10, 28, 8/3]
	my_state_inits = [[-5, 0, 30]]

	lr = FLAGS.lr # learning rate
	delta_t = FLAGS.delta_t #0.01
	tspan = np.arange(0,FLAGS.t_end,delta_t)  #np.arange(0,10000,delta_t)
	sim_model = FLAGS.model_solver
	rnn_sim_model = FLAGS.model_solver

	i = 0
	for state_init in my_state_inits:
		i += 1
		simulation_model_params = {'state_names': ['x','y','z'], 'state_init':state_init, 'delta_t':delta_t, 'smaller_delta_t': min(delta_t, delta_t), 'ode_params':(a, b, c), 'time_avg_norm':0.529, 'mxstep':0}
		assimilation_model_params = {'state_names': ['x','y','z'], 'state_init':state_init, 'delta_t':delta_t, 'smaller_delta_t': min(delta_t, delta_t), 'ode_params':(a, b, c), 'time_avg_norm':0.529, 'mxstep':0}
		all_dirs = []

		np.random.seed()

		my_random_inits = get_lorenz_inits(n=1).squeeze()

		# master output directory name
		output_dir = FLAGS.savedir + '_output' + str(i+1)

		# simulate clean and noisy data
		input_data, y_clean, _ = make_RNN_data(sim_model, tspan, simulation_model_params, noise_frac=0.05, output_dir=output_dir, drive_system=False)
		y_clean = y_clean[1:,:] # remove initial condition

		print('IGNORE the automatic way of adding noise to the data. Do it here explicitly for full control and transparency.')
		y_noisy = y_clean + eps*np.random.randn(y_clean.shape[0],y_clean.shape[1])

		# 3DVAR with perfect model
		# run_output_dir = output_dir + '/3DVAR_perfect_model'
		# torch.manual_seed(0)
		# run_3DVAR(y_clean, y_noisy, H_obs, eta, G_assim, delta_t,
		# 	sim_model, assimilation_model_params, lr,
		# 	run_output_dir, learn_assim=False, inits = my_random_inits, eps=eps)

		# # 3D VAR with perfect model + learn the assimilation matrix
		# run_output_dir = output_dir + '/3DVAR_perfect_model_learnAssimilation'
		# torch.manual_seed(0)
		# G_assim_LEARNED = run_3DVAR(y_clean, y_noisy, H_obs, eta, G_assim, delta_t,
		# 	sim_model, assimilation_model_params, lr,
		# 	run_output_dir, learn_assim=True, inits = my_random_inits, eps=eps)
		# run_output_dir = output_dir + '/3DVAR_perfect_model_learnAssimilationUSED'
		# torch.manual_seed(0)
		# run_3DVAR(y_clean, y_noisy, H_obs, eta, G_assim_LEARNED, delta_t,
		# 	sim_model, assimilation_model_params, lr,
		# 	run_output_dir, learn_assim=False, inits = my_random_inits, eps=eps)

		# ## BAD MODEL PARAMETERS NOW ##
		for eps_badness in [0]:
			assimilation_model_params = {'state_names': ['x','y','z'], 'state_init':state_init, 'delta_t':delta_t, 'smaller_delta_t': min(delta_t, delta_t), 'ode_params':(a, b*(1+eps_badness), c), 'time_avg_norm':0.529, 'mxstep':0}

			# 3D VAR with bad model + standard assimilation matrix
			run_output_dir = output_dir + '/3DVAR_epsBadness{0}'.format(eps_badness)
			torch.manual_seed(0)
			run_3DVAR(y_clean, y_noisy, eta, G_assim, delta_t,
				sim_model, assimilation_model_params, lr,
				run_output_dir, H_obs_lowfi=H_obs, learn_assim=False, inits = my_random_inits, eps=eps, opt_surfaces_only=True)

			# 3D VAR with bad model + learn assimilation matrix
			# run_output_dir = output_dir + '/3DVAR_epsBadness{0}_learnAssimilation'.format(eps_badness)
			# torch.manual_seed(0)
			# G_assim_LEARNED = run_3DVAR(y_clean, y_noisy, eta, G_assim, delta_t,
			# 	sim_model, assimilation_model_params, lr,
			# 	run_output_dir, H_obs_lowfi=H_obs, learn_assim=True, inits = my_random_inits, eps=eps, opt_surfaces_only=True)
			# run_output_dir = output_dir + '/3DVAR_epsBadness{0}_learnAssimilationUSED'.format(eps_badness)
			# torch.manual_seed(0)
			# run_3DVAR(y_clean, y_noisy, eta, G_assim_LEARNED, delta_t,
			# 	sim_model, assimilation_model_params, lr,
			# 	run_output_dir, H_obs_lowfi=H_obs, learn_assim=False, inits = my_random_inits, eps=eps, opt_surfaces_only=True)


if __name__ == '__main__':
	main()

