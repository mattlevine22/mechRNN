import argparse
import json
import numpy as np
from utils import train_chaosRNN, f_normalize_minmax
from pydoc import locate

import pdb

parser = argparse.ArgumentParser(description='mechRNN')
parser.add_argument('--settings_path', type=str, default='datagen_settings.npz', help='pathname of numpy settings dictionary')
FLAGS = parser.parse_args()

def main(settings_path=FLAGS.settings_path):

	with open(settings_path) as f:
		setts = json.load(f)

	# read in odeInstance
	odeInst = locate(setts['odeclass'])()

	# read TRAIN data
	train_set = np.load(setts['train_fname'])
	y_clean_train = train_set['y_clean']
	y_noisy_train = train_set['y_noisy']

	normz_info = {
				'Ymax': np.max(y_noisy_train, axis=0),
				'Ymin': np.min(y_noisy_train, axis=0),
				'Ymean': np.mean(y_noisy_train),
				'Ysd': np.std(y_noisy_train)
				}

	setts['y_clean_train'] = f_normalize_minmax(normz_info, y_clean_train)
	setts['y_noisy_train'] = f_normalize_minmax(normz_info, y_noisy_train)

	# read and normalize TEST data
	y_clean_test = []
	y_noisy_test = []
	y_clean_testSynch = []
	y_noisy_testSynch = []
	for fnm in setts['test_fname_list']:
		test_set = np.load(fnm)
		y_clean_test.append(f_normalize_minmax(normz_info, test_set['y_clean']))
		y_noisy_test.append(f_normalize_minmax(normz_info, test_set['y_noisy']))
		y_clean_testSynch.append(f_normalize_minmax(normz_info, test_set['y_clean_synch']))
		y_noisy_testSynch.append(f_normalize_minmax(normz_info, test_set['y_noisy_synch']))

	setts['y_clean_test'] = np.concatenate(y_clean_test)
	setts['y_noisy_test'] = np.concatenate(y_noisy_test)
	setts['y_clean_testSynch'] = np.concatenate(y_clean_testSynch)
	setts['y_noisy_testSynch'] = np.concatenate(y_noisy_testSynch)

	setts.pop('test_fname_list',None) #now remove that field

	# choose which RNN forward function to use
	try:
		setts['forward'] = locate(setts['forward'])
	except:
		setts['forward'] = None

	# pick a random initial condition
	setts['model_params']['state_init'] = odeInst.get_inits()

	# get state names
	setts['model_params']['state_names'] = odeInst.get_state_names()

	# rnn_model_params = {'state_names': state_names,
	# 					'state_init':state_init,
	# 					'delta_t':delta_t,
	# 					'ode_params':param_tuple,
	# 					'time_avg_norm':0.529,
	# 					'ode_int_method':FLAGS.ode_int_method,
	# 					'ode_int_rtol':FLAGS.ode_int_rtol,
	# 					'ode_int_atol':FLAGS.ode_int_atol,
	# 					'ode_int_max_step':np.inf}

	train_chaosRNN(**setts)
	# train_chaosRNN(forward,
	# 			y_clean_train, y_noisy_train,
	# 			y_clean_test, y_noisy_test,
	# 			y_clean_testSynch, y_noisy_testSynch,
	# 			model_params,
	# 			hidden_size=6,
	# 			n_epochs=100,
	# 			lr=0.05,
	# 			output_dir='.',
	# 			normz_info=None,
	# 			model=None,
	# 			stack_hidden=True,
	# 			stack_output=True,
	# 			precompute_model=True, kde_func=kde_scipy,
	# 			compute_kl=False,
	# 			gp_only=False,
	# 			gp_style=2,
	# 			gp_resid=True,
	# 			learn_flow = False,
	# 			alpha_list = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2],
	# 			ode_only=False):

	# train_chaosRNN(forward_chaos_hybrid_full,
	# 	y_clean_train_norm, y_noisy_train_norm,
	# 	y_clean_test_vec_norm, y_noisy_test_vec_norm,
	# 	y_clean_testSynch_vec_norm, y_noisy_testSynch_vec_norm,
	# 	rnn_BAD_model_params, hidden_size, n_epochs, lr,
	# 	run_output_dir, normz_info, rnn_sim_model,
	# 	compute_kl=FLAGS.compute_kl, alpha_list=[FLAGS.alpha],
	# 	plot_state_indices=plot_state_indices_SLOW)

if __name__ == '__main__':
	main()

