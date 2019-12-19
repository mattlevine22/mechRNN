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
parser.add_argument('--n_trains', type=int, default=1, help='number of training trajectories to summarize over')
parser.add_argument('--H_obs_lowfi', type=str2array, default=np.array([[1,0,0]]), help='low-fidelity observation operator')
parser.add_argument('--H_obs_hifi', type=str2array, default=np.array([[1,0,0],[0,1,0],[0,0,1]]), help='hi-fidelity observation operator')
parser.add_argument('--G_init_sd', type=float, default=0.1, help='standard deviation for random initialization of assimilation matrix')
parser.add_argument('--noisy_hifi', type=str2bool, default=False, help='When cheating, noisy_hifi optionally adds measurement noise to the hi-fi observations provided by H_obs_hifi.')
parser.add_argument('--optim_type', type=str, default='GradientDescent', help='NA')
# parser.add_argument('--optim_full_state', type=str2bool, default=False, help='NA')
# parser.add_argument('--n_nelder_inits', type=int, default=1, help='NA')
# parser.add_argument('--max_nelder_sols', type=int, default=None, help='Number of function calls allowed for nelder mead')
# parser.add_argument('--cheat', type=str2bool, default=False, help='cheating means using unobserved data (aka applying H_obs_hifi). If False, H_obs_hifi is inactive.')
# parser.add_argument('--new_cheat', type=str2bool, default=False, help='cheating means using unobserved data (aka applying H_obs_hifi). If False, H_obs_hifi is inactive.')
FLAGS = parser.parse_args()

# if FLAGS.optim_type is not None:
FLAGS.cheat = True
FLAGS.new_cheat = False
learn_assim = True
full_sequence = False
optim_full_state = True
# if FLAGS.optim_type=='GradientDescent':
# 	full_sequence = False
# 	optim_full_state = True
# 	FLAGS.cheat = True
# elif FLAGS.optim_type=='NelderMead':

if optim_full_state:
	state_str = 'FullState'
else:
	state_str = 'PartialState'


def main():
	eps_badness = 0
	Kmean = []
	for n_index in range(FLAGS.n_trains):
		fname = '{0}/BadModel_eps{1}_{2}_{3}_RandomInit/Train{4}/output.npz'.format(FLAGS.output_dir, FLAGS.optim_type, eps_badness, state_str, n_index)
		foo = np.load(fname)
		Klearned = foo['G_assim_history_running_mean'][-1]
		if len(Kmean)==0:
			Kmean = Klearned
		else:
			Kmean += Klearned
	Kmean = Kmean/FLAGS.n_trains
	output_savename = '{0}/BadModel_eps{1}_{2}_{3}_RandomInit/Kmean'.format(FLAGS.output_dir, FLAGS.optim_type, eps_badness, state_str)
	np.save(output_savename,Kmean)

	return Kmean



if __name__ == '__main__':
	main()

