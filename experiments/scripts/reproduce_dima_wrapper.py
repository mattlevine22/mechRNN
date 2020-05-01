import argparse
import json
from time import time
from utils import str2bool
from reproduce_dima import make_data, run_traintest
import pdb
from line_profiler import LineProfiler

parser = argparse.ArgumentParser()
parser.add_argument('--settings_path', type=str, default='datagen_settings.npz', help='pathname of numpy settings dictionary')
parser.add_argument('--profile', type=str2bool, default=False, help='option for profiling card')
parser.add_argument('--skip_datagen', type=str2bool, default=False, help='skipping the make-data step')
FLAGS = parser.parse_args()

def main(settings_path=FLAGS.settings_path, skip_datagen=FLAGS.skip_datagen):
	t0 = time()
	with open(settings_path) as f:
		setts = json.load(f)

	if skip_datagen:
		print('skipping data generation step.')
	else:
		make_data(**setts)
		print('Generated data in:', time()-t0)

	t1 = time()
	run_traintest(**setts)
	print('Ran training and made plots in:', time()-t1)

	print('Total run time:', time()-t0)

if __name__ == '__main__':
	main()

