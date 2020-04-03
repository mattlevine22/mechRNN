import os
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import seaborn as sns

import pdb

n_subsample = 10000
eps = 0.0078125
dt = 0.01

def main():
	basedir = '/groups/astuart/mlevine/writeup0/l96_dt_trials'


	results = []
	F_color = {10:'blue', 50:'green'}
	k_linestyle= ['-','--','-.',':']


	fig, (ax_list) = plt.subplots(1,1)
	ax_mean = ax_list
	# ax_std = ax_list[1]

	overall_X_min = -np.Inf
	overall_X_max = np.Inf
	for F in [10,50]:
		fname = os.path.join(basedir,'dt{dt}'.format(dt=dt),'F{F}_eps{eps}'.format(F=F,eps=eps),'TRAIN_DATA','slow_data_0_YbarData.npz')

		foo = np.load(fname)
		Ybar_data_inferred = foo['Ybar_data_inferred']
		Ybar_true = foo['Ybar_true']
		X = foo['X']

		K = X.shape[1]
		N = X.shape[0]

		if N > n_subsample:
			my_inds = np.random.choice(np.arange(N), n_subsample, replace=False)
		else:
			my_inds = np.arange(N)
		for k in range(K):
			print('Fitting GP for F={F} and k={k}'.format(F=F,k=k))
			X_k = X[my_inds,k].reshape(-1, 1)
			Ybar_k = Ybar_data_inferred[my_inds,k].reshape(-1, 1)
			gpr = GaussianProcessRegressor(alpha=1e-10).fit(X=X_k,y=Ybar_k)
			X_min = np.min(X_k)
			X_max = np.max(X_k)
			X_k_pred = np.arange(X_min,X_max,0.01).reshape(-1, 1)
			gp_mean, gp_std = gpr.predict(X_k_pred, return_std=True)

			overall_X_min = np.max((overall_X_min, X_min))
			overall_X_max = np.min((overall_X_max, X_max))

			# my_dict = {'F': F, 'k': k, 'gp_mean': gp_mean, 'gp_std': gp_std}
			# results.append(my_dict)

			ax_mean.plot(X_k_pred, gp_mean, color=F_color[F], linestyle=k_linestyle[k], label='X_{k} (F={F})'.format(k=k,F=F))
			# ax_std.plot(X_k_pred, gp_std, color=F_color[F], linestyle=k_linestyle[k], label='X_{k} (F={F})'.format(k=k,F=F))

	my_lims = (overall_X_min, overall_X_max)

	ax_mean.set_title('GP mean')
	# ax_std.set_title('GP std')
	ax_mean.set_xlim((-8,10))
	# ax_std.set_xlim((-8,10))
	ax_mean.set_ylim((-5,5))
	# ax_std.set_ylim((0,0.04))
	ax_mean.set_xlabel(r'$X_k$')
	# ax_std.set_xlabel(r'$X_k$')

	ax_mean.legend()

	fig.suptitle(r'GP-estimated closure function: $X_k \rightarrow \bar{Y}_k$')
	fig.savefig(fname=os.path.join(basedir,'dt{dt}'.format(dt=dt),'1d_GP_closure_comparison'), dpi=300)
	plt.close(fig)

if __name__ == '__main__':
	main()


