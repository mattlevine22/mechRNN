import os
import numpy as np
import pandas as pd
import argparse
from sklearn.gaussian_process import GaussianProcessRegressor
from utils import str2bool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import seaborn as sns

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--basedir', type=str, default='/groups/astuart/mlevine/writeup0/l96_dt_trials')
parser.add_argument('--infer_Ybar', type=str2bool, default=False)
parser.add_argument('--delta_t', type=float, default=0.001)
parser.add_argument('--eps', type=float, default=2**(-7))
parser.add_argument('--n_subsample', type=int, default=1000)
FLAGS = parser.parse_args()


def main(basedir=FLAGS.basedir,
	infer_Ybar=FLAGS.infer_Ybar,
	dt=FLAGS.delta_t,
	eps=FLAGS.eps,
	n_subsample=FLAGS.n_subsample):

	results = []
	F_color = {10:'blue', 50:'green'}
	k_linestyle= ['-','--','-.',':']

	# for alpha in [1, 1e-1, 1e-5, 1e-10]:
	for alpha in [100, 50, 25, 10, 1]:
		fig, (ax_list) = plt.subplots(1,1)
		ax_mean = ax_list
		# ax_std = ax_list[1]

		overall_X_min = -np.Inf
		overall_X_max = np.Inf
		for F in [50]:
			fname = os.path.join(basedir,'dt{dt}'.format(dt=dt),'F{F}_eps{eps}'.format(F=F,eps=eps),'TRAIN_DATA','slow_data_0_YbarData.npz')

			foo = np.load(fname)
			Ybar_data_inferred = foo['Ybar_data_inferred']
			Ybar_true = foo['Ybar_true']
			X = foo['X']

			K = X.shape[1]
			# N = X.shape[0]

			# if N > n_subsample:
			# 	my_inds = np.random.choice(np.arange(N), n_subsample, replace=False)
			# else:
			# 	my_inds = np.arange(N)
			# for k in range(K):
			# 	print('Fitting GP for F={F} and k={k}'.format(F=F,k=k))
			# 	X_k = X[my_inds,k].reshape(-1, 1)
			# 	if infer_Ybar:
			# 		Ybar_k = Ybar_data_inferred[my_inds,k].reshape(-1, 1)
			# 	else:
			# 		Ybar_k = Ybar_true[my_inds,k].reshape(-1, 1)
			# 	gpr = GaussianProcessRegressor(alpha=alpha, n_restarts_optimizer=15).fit(X=X_k,y=Ybar_k)
			# 	X_min = np.min(X_k)
			# 	X_max = np.max(X_k)
			# 	X_k_pred = np.arange(X_min,X_max,0.01).reshape(-1, 1)
			# 	gp_mean, gp_std = gpr.predict(X_k_pred, return_std=True)

			# 	overall_X_min = np.max((overall_X_min, X_min))
			# 	overall_X_max = np.min((overall_X_max, X_max))

			# 	# my_dict = {'F': F, 'k': k, 'gp_mean': gp_mean, 'gp_std': gp_std}
			# 	# results.append(my_dict)

			# 	ax_mean.plot(X_k_pred, gp_mean, color=F_color[F], linestyle=':', label='X_{k} (F={F})'.format(k=k,F=F))
				# ax_std.plot(X_k_pred, gp_std, color=F_color[F], linestyle=k_linestyle[k], label='X_{k} (F={F})'.format(k=k,F=F))

			# fit GP to all states together
			Xtrain = X.reshape(-1, 1)
			X_min = np.min(Xtrain)
			X_max = np.max(Xtrain)
			if infer_Ybar:
				ytrain = Ybar_data_inferred.reshape(-1, 1)
			else:
				ytrain = Ybar_true.reshape(-1, 1)
			N = Xtrain.shape[0]
			if N > n_subsample:
				my_inds = np.random.choice(np.arange(N), n_subsample, replace=False)
			else:
				my_inds = np.arange(N)

			gpr = GaussianProcessRegressor(alpha=alpha, n_restarts_optimizer=15).fit(X=Xtrain[my_inds],y=ytrain[my_inds])
			X_k_pred = np.arange(X_min,X_max,0.01).reshape(-1, 1)
			gp_mean, gp_std = gpr.predict(X_k_pred, return_std=True)
			ax_mean.scatter(Xtrain,ytrain, s=5, color='gray', alpha=0.8, label='Data')
			ax_mean.scatter(Xtrain[my_inds],ytrain[my_inds], s=5, color='red', label='Training')
			ax_mean.plot(X_k_pred, gp_mean, color=F_color[F], linestyle='-', label='X-all (F={F})'.format(F=F))

		# my_lims = (overall_X_min, overall_X_max)

		ax_mean.set_title('GP mean')
		# ax_std.set_title('GP std')
		# ax_mean.set_xlim((-8,10))
		# ax_std.set_xlim((-8,10))
		# ax_mean.set_ylim((-5,5))
		# ax_std.set_ylim((0,0.04))
		ax_mean.set_xlabel(r'$X_k$')
		# ax_std.set_xlabel(r'$X_k$')

		ax_mean.legend()

		fig.suptitle(r'GP-estimated closure function: $X_k \rightarrow \bar{Y}_k$')
		fig.savefig(fname=os.path.join(basedir,'dt{dt}'.format(dt=dt),'1d_F50_GP_closure_n{n_subsample}_inferYbar{infer_Ybar}_comparison_alpha{alpha}.png'.format(infer_Ybar=infer_Ybar, n_subsample=n_subsample, alpha=alpha)), dpi=300)
		plt.close(fig)

if __name__ == '__main__':
	main()


