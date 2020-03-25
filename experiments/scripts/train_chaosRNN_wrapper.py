import argparse
import json
import numpy as np
from utils import train_chaosRNN, f_normalize_minmax
from pydoc import locate
from time import time

import pdb

parser = argparse.ArgumentParser(description='mechRNN')
parser.add_argument('--settings_path', type=str, default='datagen_settings.npz', help='pathname of numpy settings dictionary')
FLAGS = parser.parse_args()

def main(settings_path=FLAGS.settings_path):
	t0 = time()
	with open(settings_path) as f:
		setts = json.load(f)

	# read in odeInstance
	odeInst = locate(setts['odeclass'])(**setts['param_dict'])

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

	setts['normz_info'] = normz_info

	setts['y_clean_train'] = f_normalize_minmax(normz_info, y_clean_train)
	setts['y_noisy_train'] = f_normalize_minmax(normz_info, y_noisy_train)
	# setts['y_clean_trainSynch'] = f_normalize_minmax(normz_info, train_set['y_clean_synch'])
	# setts['y_noisy_trainSynch'] = f_normalize_minmax(normz_info, train_set['y_noisy_synch'])

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

	setts['y_clean_test'] = np.stack(y_clean_test)
	setts['y_noisy_test'] = np.stack(y_noisy_test)
	setts['y_clean_testSynch'] = np.stack(y_clean_testSynch)
	setts['y_noisy_testSynch'] = np.stack(y_noisy_testSynch)


	# check for fast test data and read that in
	if 'test_fast_fname_list' in setts and setts['test_fast_fname_list']:
		y_fast_test = []
		for fnm in setts['test_fast_fname_list']:
			test_set = np.load(fnm)
			y_fast_test.append(test_set['y_clean'])
		setts['y_fast_test'] = np.stack(y_fast_test)
		setts.pop('test_fast_fname_list',None) #now remove that field

	# try to read in fast train data
	try:
		setts['y_fast_train'] = np.load(setts['train_fast_fname'])['y_clean']
	except:
		print('Couldnt read and save fast train data, skipping that part.')
		pass

	try:
		setts.pop('train_fast_fname',None) #now remove that field
	except:
		# not a key
		pass

	setts.pop('test_fname_list',None) #now remove that field
	setts.pop('train_fname',None) #now remove that field
	setts.pop('odeclass',None) #now remove that field
	setts.pop('param_dict',None) #now remove that field


	# choose which RNN forward function to use
	try:
		setts['forward'] = locate(setts['forward'])
	except:
		setts['forward'] = None

	# pick a random initial condition
	# setts['model_params']['state_init'] = odeInst.get_inits()
	# get state names
	setts['model_params']['state_names'] = odeInst.get_state_names()

	setts['model'] = odeInst.rhs

	setts['plot_state_indices'] = odeInst.plot_state_indices()

	setts['ODE'] = odeInst

	train_chaosRNN(**setts)
	print('Ran training in:', time()-t0)

if __name__ == '__main__':
	main()

