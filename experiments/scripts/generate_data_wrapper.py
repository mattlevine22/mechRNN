import argparse
import json
from utils import generate_data
from pydoc import locate
from time import time

import pdb

parser = argparse.ArgumentParser(description='mechRNN')
parser.add_argument('--settings_path', type=str, default='datagen_settings.json', help='pathname of numpy settings dictionary')
parser.add_argument('--slow_name', type=str, default='slow_data.npz', help='pathname of output data')
parser.add_argument('--fast_name', type=str, default='fast_data.npz', help='pathname of output data')
FLAGS = parser.parse_args()

def main(settings_path=FLAGS.settings_path, output_path=FLAGS.output_path):
	with open(settings_path) as f:
	  setts = json.load(f)
	# https://stackoverflow.com/questions/547829/how-to-dynamically-load-a-python-class
	my_class = locate(setts['odeclass']) #e.g. 'odelibrary.L96M'
	setts['ODE'] = my_class(**setts['param_dict'])
	setts.pop('param_dict',None)
	setts.pop('odeclass',None)
	t0 = time()
	generate_data(output_path=output_path, **setts)
	print('Generated data in:', time()-t0)

if __name__ == '__main__':
	main()

