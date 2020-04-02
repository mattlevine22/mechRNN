import os
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import seaborn as sns

def main():
	basedir = '/groups/astuart/mlevine/writeup0/l96_dt_trials'

	eps = 0.0078125
	dt = 0.01

	results = []
	F_color = {10:'blue', 50:'green'}
	k_linestyle= ['-','--','-.',':']


	fig, (ax_list) = plt.subplots(1,2)
	ax_mean = ax_list[0]
	ax_std = ax_list[1]

	for F in [10,50]:
		fname = os.path.join(basedir,'dt{dt}'.format(dt=dt),'F{F}_eps{eps}'.format(F=F,eps=eps),'TRAIN_DATA','slow_data_0_YbarData.npz')

		foo = np.load(fname)
		Ybar_data_inferred = foo['Ybar_data_inferred']
		Ybar_true = foo['Ybar_true']
		X = foo['X']

		K = X.shape[1]
		for k in range(K):
			X_k = X[:,k].reshape(-1, 1)
			Ybar_k = Ybar_data_inferred[:,k].reshape(-1, 1)
			gpr = GaussianProcessRegressor(alpha=1e-10).fit(X=X_k,y=Ybar_k)
			X_min = np.min(X_k)
			X_max = np.max(X_k)
			X_k_pred = np.arange(X_min,X_max,0.01).array.reshape(-1, 1)
			gp_mean, gp_std = gpr.predict(X_k_pred, return_std=True)

			# my_dict = {'F': F, 'k': k, 'gp_mean': gp_mean, 'gp_std': gp_std}
			# results.append(my_dict)

			ax_mean.plot(X_k, gp_mean, color=F_color[F], linestyle=k_linestyle[k], label='X_{k} (F={F})'.format(k=k,F=F))
			ax_std.plot(X_k, gp_std, color=F_color[F], linestyle=k_linestyle[k], label='X_{k} (F={F})'.format(k=k,F=F))


	ax_mean.set_title('GP mean')
	ax_std.set_title('GP std')

	ax_mean.legend()

	fig.suptitle(r'GP-estimated closure function: $X_k \rightarrow \bar{Y}_k$')
	fig.savefig(fname=os.path.join(basedir,'dt{dt}'.format(dt=dt),'1d_GP_closure_comparison'))
	plt.close(fig)

if __name__ == '__main__':
	main()


