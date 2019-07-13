import sys
import os
from pathlib import Path
from utils import extract_delta_t_performance

def main(pathname):
	# pathname = '/Users/matthewlevine/code_projects/mechRNN/experiments/June4/lorenz63_eps_loop_10000epochs_v2'
	my_dirs = []
	for x in Path(pathname).glob('**/*RNN*'):
		x = str(x)
		if os.path.exists(x+'/rnn_fit_ode_TEST.png') or os.path.exists(x+'/test_fit_ode.png'):
			my_dirs.append(x)
	for x in Path(pathname).glob('**/*hybrid*'):
		x = str(x)
		if os.path.exists(x+'/rnn_fit_ode_TEST.png') or os.path.exists(x+'/test_fit_ode.png') or os.path.exists(x+'/fit_ode_TEST.png'):
			my_dirs.append(x)

	extract_delta_t_performance(my_dirs,output_fname=pathname+'/compare_delta_t')

if __name__ == '__main__':
	main(sys.argv[1])
