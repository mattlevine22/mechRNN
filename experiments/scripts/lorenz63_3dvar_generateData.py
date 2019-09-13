from utils import make_RNN_data, get_lorenz_inits, lorenz63
import numpy as np
import argparse
import os
import pdb

parser = argparse.ArgumentParser(description='3DVAR')
parser.add_argument('--eps', type=float, default=1, help='observation noise coefficient')
parser.add_argument('--delta_t', type=float, default=0.01, help='time step of simulation')
parser.add_argument('--t_end', type=float, default=500, help='length of simulation')
parser.add_argument('--model_solver', default=lorenz63, help='ode function')
parser.add_argument('--output_dir', type=str, default='default_output', help='filename for generated data')
parser.add_argument('--output_filename', type=str, default='training_data', help='filename for generated data')
parser.add_argument('--n_trajectories', type=int, default=2, help='number of trajectories to generate')

FLAGS = parser.parse_args()

def main():

	if not os.path.exists(FLAGS.output_dir):
		os.makedirs(FLAGS.output_dir)

	output_full_fname = '{0}/{1}'.format(FLAGS.output_dir, FLAGS.output_filename)
	np.random.seed()

	n_trajectories = FLAGS.n_trajectories

	(a,b,c) = [10, 28, 8/3]
	delta_t = FLAGS.delta_t #0.01
	tspan = np.arange(0,FLAGS.t_end,delta_t)  #np.arange(0,10000,delta_t)
	sim_model = FLAGS.model_solver

	true_state_init = get_lorenz_inits(n=n_trajectories)

	y_clean_ALL = np.zeros((n_trajectories,len(tspan)-1,true_state_init.shape[1]))
	y_noisy_ALL = np.zeros((n_trajectories,len(tspan)-1,true_state_init.shape[1]))

	for n in range(n_trajectories):
		simulation_model_params = {'state_names': ['x','y','z'], 'state_init':true_state_init[n], 'delta_t':delta_t, 'smaller_delta_t': min(delta_t, delta_t), 'ode_params':(a, b, c), 'time_avg_norm':0.529, 'mxstep':0}
		# assimilation_model_params = {'state_names': ['x','y','z'], 'state_init':state_init, 'delta_t':delta_t, 'smaller_delta_t': min(delta_t, delta_t), 'ode_params':(a, b, c), 'time_avg_norm':0.529, 'mxstep':0}
		# all_dirs = []

		# simulate clean and noisy data
		input_data, y_clean, _ = make_RNN_data(sim_model, tspan, simulation_model_params, noise_frac=0.05, drive_system=False)
		y_clean = y_clean[1:,:] # remove initial condition
		# print('IGNORE the automatic way of adding noise to the data. Do it here explicitly for full control and transparency.')
		y_noisy = y_clean + FLAGS.eps*np.random.randn(y_clean.shape[0],y_clean.shape[1])

		y_clean_ALL[n,:,:] = y_clean[:,:]
		y_noisy_ALL[n,:,:] = y_noisy[:,:]

	np.savez(output_full_fname, y_clean=y_clean_ALL, y_noisy=y_noisy_ALL, true_state_init=true_state_init, random_state_init=get_lorenz_inits(n=n_trajectories), model_params=simulation_model_params)


if __name__ == '__main__':
	main()

