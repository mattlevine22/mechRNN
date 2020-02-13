import sys
import os
from pathlib import Path
from summarizers import extract_epsilon_performance
from utils import str2bool
import argparse

parser = argparse.ArgumentParser(description='mechRNN')
parser.add_argument('--ignore_flow', type=str2bool, default=True, help='if True, does not plot flow-based learning results')
parser.add_argument('--pathname', type=str, default='.', help='directory with results to read in/out')
FLAGS = parser.parse_args()

def main():
	pathname = FLAGS.pathname
	# pathname = '/Users/matthewlevine/code_projects/mechRNN/experiments/June4/lorenz63_eps_loop_10000epochs_v2'
	my_dirs = []
	for x in Path(pathname).glob('**/*'):
		x = str(x)
		if os.path.exists(x+'/loss_vec_clean_test.txt'):
			my_dirs.append(x)
	# for x in Path(pathname).glob('**/*hybrid*'):
	# 	x = str(x)
	# 	if os.path.exists(x+'/rnn_fit_ode_TEST.png') or os.path.exists(x+'/test_fit_ode.png'):
	# 		my_dirs.append(x)
	if len(my_dirs):
		extract_epsilon_performance(my_dirs,output_fname=pathname+'/compare_epsilons',ignore_flow=FLAGS.ignore_flow)

if __name__ == '__main__':
	main()
