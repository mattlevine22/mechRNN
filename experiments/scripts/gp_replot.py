import os
import numpy as np
import pandas as pd
import pickle
import argparse
import glob
from sklearn.gaussian_process import GaussianProcessRegressor
from utils import str2bool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import seaborn as sns

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--infer_Ybar', type=str2bool, default=False)
parser.add_argument('--delta_t', type=float, default=0.001)
parser.add_argument('--eps', type=float, default=2**(-7))
parser.add_argument('--n_subsample', type=int, default=1000)
FLAGS = parser.parse_args()


def plot_gp_x_vs_y(gp_path, data_path, infer_Ybar=False):
	# this compares the GP prediction over Xk VS the true Ybark
	# read in training data
	foo = np.load(data_path)
	Ybar_data_inferred = foo['Ybar_data_inferred']
	Ybar_true = foo['Ybar_true']
	Xtrain= foo['X'].reshape(-1, 1)
	if infer_Ybar:
		ytrain = Ybar_data_inferred.reshape(-1, 1)
	else:
		ytrain = Ybar_true.reshape(-1, 1)

	# initialize plot
	fig, (ax_list) = plt.subplots(1,1)
	ax_mean = ax_list
	ax_mean.scatter(Xtrain,ytrain, s=5, color='gray', alpha=0.8, label='Data')
	# ax_mean.scatter(Xtrain[my_inds],ytrain[my_inds], s=5, color='red', label='Training')

	# Now, load and plot GPRs
	X_min = np.min(Xtrain)
	X_max = np.max(Xtrain)
	X_k_pred = np.arange(X_min,X_max,0.01).reshape(-1, 1)
	gpr_list = pickle.load(open(gp_path,'rb'))
	for gpr in gpr_list:
		gp_mean, gp_std = gpr.predict(X_k_pred, return_std=True)
		ax_mean.plot(X_k_pred, gp_mean, color='black', linestyle='-')

	ax_mean.set_title('GP mean')
	ax_mean.set_xlabel(r'$X_k$')
	ax_mean.set_ylabel(r'$\bar{Y}_k$')
	# ax_mean.legend()

	output_path = os.path.join(os.path.split(gp_path)[0],'gp_Xk_vs_barYk.png')
	fig.savefig(fname=output_path, dpi=300)
	plt.close(fig)


def main():
	base_path = '/groups/astuart/mlevine/writeup0/l96_dt_trials_v2'
	eps = 0.0078125
	for dt in [1e-3, 1e-4, 1e-2]:
		for F in [10, 50]:
			run_dir = os.path.join(base_path, 'dt{dt}/F{F}_eps{eps}'.format(dt=dt,F=F,eps=eps))
			data_path = os.path.join(run_dir, 'TRAIN_DATA','slow_data_0_YbarData.npz')
			glob_str = os.path.join(run_dir,'Init0/*/gpr_list.p')
			for gp_path in glob.glob(glob_str):
				# this compares the GP prediction over Xk VS the true Ybark
				plot_gp_x_vs_y(gp_path=gp_path, data_path=data_path)


if __name__ == '__main__':
	main()


