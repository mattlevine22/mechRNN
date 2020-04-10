import os
import json
import numpy as np
import argparse
from time import time
from utils import mkdir_p
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from odelibrary import L96M
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--testing_fname', type=str, default='/groups/astuart/mlevine/writeup0/l96_dt_trials_v2/dt0.001/F50_eps0.0078125/reproduce_dima/testing.npz')
parser.add_argument('--training_fname', type=str, default='/groups/astuart/mlevine/writeup0/l96_dt_trials_v2/dt0.001/F50_eps0.0078125/reproduce_dima/training.npz')
parser.add_argument('--dima_data_path', type=str, default='/home/mlevine/mechRNN/experiments/scripts/dima_gp_training_data.npy')
parser.add_argument('--output_dir', type=str, default='/groups/astuart/mlevine/writeup0/l96_dt_trials_v2/dt0.001/F50_eps0.0078125/reproduce_dima/')
FLAGS = parser.parse_args()


def get_inds(N_total, N_subsample):
	if N_total > N_subsample:
		my_inds = np.random.choice(np.arange(N_total), N_subsample, replace=False)
	else:
		my_inds = np.arange(N_total)
	return my_inds

def kde_scipy(x, x_grid):
	"""Kernel Density Estimation with Scipy"""
	# Note that scipy weights its bandwidth by the covariance of the
	# input data.  To make the results comparable to the other methods,
	# we divide the bandwidth by the sample standard deviation here.
	kde = gaussian_kde(x)
	return kde.evaluate(x_grid)


def eliminate_dima(
		testing_fname=FLAGS.testing_fname,
		training_fname=FLAGS.training_fname,
		output_dir=FLAGS.output_dir,
		K=9,
		J=8,
		F=10,
		eps=2**(-7),
		ode_int_method='RK45',
		ode_int_atol=1e-6,
		ode_int_rtol=1e-3,
		ode_int_max_step=1e-3,
		delta_t = 1e-3,
		t_synch = 5,
		t_train = 10,
		t_invariant_measure = 10,
		n_subsample_gp = 800,
		n_subsample_kde=int(1e6),
		alpha_list = [0.5,1,0.1]):

	mkdir_p(output_dir)

	# initialize ode object
	ODE = L96M(K=K, J=J, F=F, eps=eps)
	ODE.set_stencil() # this is a default, empty usage that is required

	# Get training data:
	t_eval = np.arange(0, (t_synch+t_train), delta_t)
	# Run for 500+T and output solutions at dT
	t0 = time()
	y0 = ODE.get_inits().squeeze()
	sol = solve_ivp(fun=lambda t, y: ODE.rhs(y, t), t_span=(t_eval[0], t_eval[-1]), y0=y0, method=ode_int_method, rtol=ode_int_rtol, atol=ode_int_atol, max_step=ode_int_max_step, t_eval=t_eval)
	y_clean = sol.y.T
	print('Generated Training data in:', (time()-t0)/60,'minutes')

	# Take last T-traj as training data
	ntsynch = int(t_synch/delta_t)
	X_train = y_clean[ntsynch:,:K] 	# Take :K as slow-training
	y_fast = y_clean[ntsynch:-1,K:] # Use fast K: to compute Ybar-true (skipping last time-index since we can't use data to infer that)
	# Compute Ybar-true directly
	Ybar_true = y_fast.reshape( (y_fast.shape[0], ODE.J, ODE.K), order = 'F').sum(axis = 1) / ODE.J
	# Use slow-training to compute Ybar-infer (note we can't get Ybar_k for the last time index)
	Ybar_data_inferred = ODE.implied_Ybar(X_in=X_train[:-1,:], X_out=X_train[1:,:], delta_t=delta_t)

	np.savez(training_fname, X_train=X_train, Ybar_true=Ybar_true, Ybar_data_inferred=Ybar_data_inferred)

	# Get data for estimating the true invariant measure:
	# Run for 5k and output solutions at dT
	t_eval = np.arange(0, (t_synch+t_invariant_measure), delta_t)
	y0 = ODE.get_inits().squeeze()
	t0 = time()
	sol = solve_ivp(fun=lambda t, y: ODE.rhs(y, t), t_span=(t_eval[0], t_eval[-1]), y0=y0, method=ode_int_method, rtol=ode_int_rtol, atol=ode_int_atol, max_step=ode_int_max_step, t_eval=t_eval)
	y_clean = sol.y.T
	print('Generated invariant measure Testing data in:', (time()-t0)/60,'minutes')

	# Remove first 500 and plot KDE
	X_test = y_clean[ntsynch:,:K].reshape(-1, 1)
	x_grid = np.linspace(min(X_test), max(X_test), 1000)
	np.savez(testing_fname, X_test=X_test, ntsynch=ntsynch, t_eval=t_eval, x_grid=x_grid, y0=y0, K=K)

	return

def plot_data(testing_fname=FLAGS.testing_fname,
	training_fname=FLAGS.training_fname,
	dima_data_path=FLAGS.dima_data_path,
	output_dir = FLAGS.output_dir,
	n_subsample_gp=800,
	n_subsample_kde=int(1e6),
	n_restarts_optimizer=15,
	K=9,
	J=8,
	F=10,
	eps=2**(-7),
	ode_int_method='RK45',
	ode_int_atol=1e-6,
	ode_int_rtol=1e-3,
	ode_int_max_step=1e-3,
	alpha_list = [0.5,1]):

	# output dir
	output_fname = os.path.join(output_dir,'eliminate_dima.png')

	# initialize ode object
	ODE = L96M(K=K, J=J, F=F, eps=eps)
	ODE.set_stencil() # this is a default, empty usage that is required


	# get initial colors
	prop_cycle = plt.rcParams['axes.prop_cycle']
	color_list = prop_cycle.by_key()['color']

	foo = np.load(training_fname)
	goo = np.load(testing_fname)

	# Fit GPRs to the data
	X = foo['X_train'][:-1,:].reshape(-1,1)
	Y_true = foo['Ybar_true'].reshape(-1,1)
	Y_inferred = foo['Ybar_data_inferred'].reshape(-1,1)
	train_inds = get_inds(N_total=X.shape[0], N_subsample=n_subsample_gp)
	my_lims = (-6, 13)
	X_pred = np.arange(my_lims[0],my_lims[1],0.01).reshape(-1, 1)

	X_test = goo['X_test']
	x_grid = goo['x_grid']
	t_eval = goo['t_eval']
	ntsynch = goo['ntsynch']
	K = goo['K']
	y0 = goo['y0'][:K]
	test_inds = get_inds(N_total=X_test.shape[0], N_subsample=n_subsample_kde)
	fig, (ax_gp, ax_kde) = plt.subplots(1,2,figsize=[8,4])
	t0 = time()
	sns.kdeplot(X_test[test_inds].squeeze(), ax=ax_kde, label='Matt-True', color='black', linestyle='--')
	print('Plotted matt-KDE of invariant measure in:', (time()-t0)/60,'minutes')

	# plot Dimas data invariant measure
	X_dima = np.load(dima_data_path)
	sns.kdeplot(X_dima[:,0].squeeze(), ax=ax_kde, label='Dima-True', color='black', linestyle=':')
	print('Plotted dima-KDE of invariant measure in:', (time()-t0)/60,'minutes')

	ax_kde.legend()
	fig.savefig(fname=output_fname)

	# plot training data
	ax_gp.scatter(X, Y_true, s=5, color='gray', alpha=0.8, label='Data')
	ax_gp.scatter(X[train_inds], Y_true[train_inds], s=5, color='red', alpha=0.8, label='Training Data')

	# check null predictor (hy*x)
	# ODE.set_G0_predictor()
	# g0_mean = ODE.hy*X_pred
	# ax_gp.plot(X_pred, g0_mean, color='black', linestyle='--', label='G0')

	# t0 = time()
	# sol = solve_ivp(fun=lambda t, y: ODE.regressed(y, t), t_span=(t_eval[0], t_eval[-1]), y0=y0, method=ode_int_method, rtol=ode_int_rtol, atol=ode_int_atol, max_step=ode_int_max_step, t_eval=t_eval)
	# y_clean = sol.y.T
	# X_test_G0 = y_clean[ntsynch:,:K].reshape(-1, 1)
	# print('Generated invariant measure for G0:', (time()-t0)/60,'minutes')
	# sns.kdeplot(X_test_G0[test_inds].squeeze(), ax=ax_kde, color='gray', linestyle='-', label='G0')

	# ax_kde.plot(x_grid, kde_scipy(x=X_test, x_grid=x_grid), label='True')
	for c in range(len(alpha_list)):
		alpha = alpha_list[c]
		color = color_list[c]
		print('alpha=',alpha, color)

		# intialize gpr
		GP_ker = 1.0 * RBF(3, (1e-10, 1e+6))
		my_gpr = GaussianProcessRegressor(
			kernel = GP_ker,
			n_restarts_optimizer = n_restarts_optimizer,
			alpha = alpha
		)

		# fit GPR-Ybartrue to Xk vs Ybar-true
		gpr_true = my_gpr.fit(X=X[train_inds], y=Y_true[train_inds])
		gpr_true_mean = gpr_true.predict(X_pred, return_std=False)
		# now run gp-corrected ODE
		ODE.set_predictor(gpr_true.predict)
		t0 = time()
		sol = solve_ivp(fun=lambda t, y: ODE.regressed(y, t), t_span=(t_eval[0], t_eval[-1]), y0=y0, method=ode_int_method, rtol=ode_int_rtol, atol=ode_int_atol, max_step=ode_int_max_step, t_eval=t_eval)
		y_clean = sol.y.T
		X_test_gpr_true = y_clean[ntsynch:,:K].reshape(-1, 1)
		print('Generated invariant measure for GP-true:', (time()-t0)/60,'minutes')

		# fit GPR-Ybarpprox to Xk vs Ybar-infer
		gpr_approx = my_gpr.fit(X=X[train_inds], y=Y_inferred[train_inds])
		gpr_approx_mean = gpr_approx.predict(X_pred, return_std=False)
		# now run gp-corrected ODE
		ODE.set_predictor(gpr_approx.predict)
		t0 = time()
		sol = solve_ivp(fun=lambda t, y: ODE.regressed(y, t), t_span=(t_eval[0], t_eval[-1]), y0=y0, method=ode_int_method, rtol=ode_int_rtol, atol=ode_int_atol, max_step=ode_int_max_step, t_eval=t_eval)
		y_clean = sol.y.T
		X_test_gpr_approx = y_clean[ntsynch:,:K].reshape(-1, 1)
		print('Generated invariant measure for GP-true:', (time()-t0)/60,'minutes')

		# Plot each GPR along X vs Ybar (give alpha a color; give True/Infer a line-style)
		ax_gp.plot(X_pred, gpr_true_mean, color=color, linestyle='--', label='X_k vs true Y-avg (alpha={alpha})'.format(alpha=alpha))
		ax_gp.plot(X_pred, gpr_approx_mean, color=color, linestyle='-', label='X_k vs inferred Y-avg (alpha={alpha})'.format(alpha=alpha))
		ax_gp.set_xlabel(r'$X_k$')
		ax_gp.set_ylabel(r'$\bar{Y}_k$')
		# ax_gp.legend()

		# Test each of these GPRs for their ability to reproduce invariant measure:
		sns.kdeplot(X_test_gpr_true[test_inds].squeeze(), ax=ax_kde, color=color, linestyle='--', label='Ybar-True (alpha={alpha})'.format(alpha=alpha))
		sns.kdeplot(X_test_gpr_approx[test_inds].squeeze(), ax=ax_kde, color=color, linestyle='-', label='Ybar-Inferred (alpha={alpha})'.format(alpha=alpha))
		ax_kde.set_xlabel(r'$X_k$')
		# ax_kde.set_ylabel(r'$\bar{Y}_k')
		# ax_kde.legend()
		ax_kde.legend().set_visible(False)
		# save figure after each loop
		fig.savefig(fname=output_fname)
	# close the fig when you're done!
	plt.close(fig)
	return


if __name__ == '__main__':
	eliminate_dima()
	plot_data()


