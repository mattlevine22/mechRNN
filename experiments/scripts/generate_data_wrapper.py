import argparse
import json
from utils import generate_data
from pydoc import locate

import pdb

parser = argparse.ArgumentParser(description='mechRNN')
parser.add_argument('--settings_path', type=str, default='datagen_settings.npz', help='pathname of numpy settings dictionary')
FLAGS = parser.parse_args()

def main(settings_path=FLAGS.settings_path):
	with open(settings_path) as f:
	  setts = json.load(f)
	# https://stackoverflow.com/questions/547829/how-to-dynamically-load-a-python-class
	pdb.set_trace()
	my_class = locate(setts['odeclass']) #e.g. 'odelibrary.L96M'
	setts['ODE'] = my_class(**setts['param_dict'])
	setts.pop('param_dict',None)
	setts.pop('odeclass',None)
	generate_data(**setts)

if __name__ == '__main__':
	main()

