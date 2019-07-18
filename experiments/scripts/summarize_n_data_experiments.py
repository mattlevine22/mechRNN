import sys
import os
from pathlib import Path
from utils import extract_n_data_performance

def main(pathname):
	# pathname = '/Users/matthewlevine/code_projects/mechRNN/experiments/June4/lorenz63_eps_loop_10000epochs_v2'
	my_dirs = []
	for x in Path(pathname).glob('**/*RNN*'):
		x = str(x)
		if os.path.exists(x+'/rnn_fit_ode_TEST_0.png') or os.path.exists(x+'/test_fit_ode_0.png'):
			my_dirs.append(x)
	for x in Path(pathname).glob('**/*hybrid*'):
		x = str(x)
		if os.path.exists(x+'/rnn_fit_ode_TEST_0.png') or os.path.exists(x+'/test_fit_ode.png') or os.path.exists(x+'/fit_ode_TEST_0.png'):
			my_dirs.append(x)

	if len(my_dirs):
		extract_n_data_performance(my_dirs,output_fname=pathname+'/compare_n_training_points')

if __name__ == '__main__':
	main(sys.argv[1])
