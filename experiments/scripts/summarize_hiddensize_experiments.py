import sys
import os
from pathlib import Path
from summarizers import extract_hidden_size_performance

def main(pathname):
	my_dirs = []
	for x in Path(pathname).glob('**/*RNN*'):
		x = str(x)
		if os.path.exists(x+'/loss_vec_clean_test.txt'):
			my_dirs.append(x)
	extract_hidden_size_performance(my_dirs,output_fname=pathname+'/compare_hidden_sizes')

if __name__ == '__main__':
	main(sys.argv[1])
