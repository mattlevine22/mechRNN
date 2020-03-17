import argparse
import json
from utils import generate_data
from pydoc import locate

parser = argparse.ArgumentParser(description='mechRNN')
parser.add_argument('--settings_path', type=str, default='datagen_settings.npz', help='pathname of numpy settings dictionary')
FLAGS = parser.parse_args()

def main(settings_path=FLAGS.settings_path):
	with open(settings_path) as f:
	  setts = json.load(f)
	# https://stackoverflow.com/questions/547829/how-to-dynamically-load-a-python-class
	pdb.set_trace()
	my_class = locate(setts['odeclass']) #e.g. 'odelibrary.L96M'
	my_class = my_class(**setts['param_dict'])
	setts['ODE'] = my_class(**setts['param_dict'])
	# setts['f_get_inits'] = getattr(my_class, 'get_inits')
	generate_data(**setts)
	return

if __name__ == '__main__':
	main()

