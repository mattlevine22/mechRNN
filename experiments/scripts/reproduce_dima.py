import os
import json
import numpy as np
import argparse
from time import time
from utils import mkdir_p
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from odelibrary import L96M
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--delta_t', type=float, default=1e-3, help='Data sampling rate')
parser.add_argument('--K', type=int, default=9, help='number of slow variables')
parser.add_argument('--J', type=int, default=8, help='number of fast variables coupled to a single slow variable')
parser.add_argument('--F', type=float, default=10)
parser.add_argument('--eps', type=float, default=2**(-7))
parser.add_argument('--ode_int_method', type=str, default='RK45', help='See scipy solve_ivp documentation for options.')
parser.add_argument('--ode_int_atol', type=float, default=1e-6, help='This is a much higher-fidelity tolerance than defaults for solve_ivp')
parser.add_argument('--ode_int_rtol', type=float, default=1e-3, help='This is a much higher-fidelity tolerance than defaults for solve_ivp')
parser.add_argument('--ode_int_max_step', type=float, default=1e-3, help='This is a much higher-fidelity tolerance than defaults for solve_ivp')
parser.add_argument('--rng_seed', type=float, default=63)
parser.add_argument('--t_synch', type=float, default=5)
parser.add_argument('--t_train', type=float, default=10)
parser.add_argument('--t_invariant_measure', type=float, default=10)
parser.add_argument('--n_subsample_gp', type=int, default=800)
parser.add_argument('--n_subsample_kde', type=int, default=int(1e9))
parser.add_argument('--n_restarts_optimizer', type=int, default=15)
parser.add_argument('--testing_fname', type=str, default='testing.npz')
parser.add_argument('--training_fname', type=str, default='training.npz')
parser.add_argument('--dima_data_path', type=str, default='/home/mlevine/mechRNN/experiments/scripts/dima_gp_training_data.npy')
parser.add_argument('--output_dir', type=str, default='/groups/astuart/mlevine/writeup0/l96_dt_trials_v2/dt0.001/F50_eps0.0078125/reproduce_dima/')
FLAGS = parser.parse_args()

# python3 reproduce_dima.py --testing_fname testing1.npz --training_fname training1.npz --dima_data_path /Users/matthewlevine/code_projects/mechRNN/experiments/scripts/dima_gp_training_data.npy --output_dir .

def get_inds(N_total, N_subsample):
	if N_total > N_subsample:
		my_inds = np.random.choice(np.arange(N_total), N_subsample, replace=False)
	else:
		my_inds = np.arange(N_total)
	return my_inds

def eliminate_dima(
	testing_fname=os.path.join(FLAGS.output_dir,FLAGS.testing_fname),
	training_fname=os.path.join(FLAGS.output_dir,FLAGS.training_fname),
	output_dir=FLAGS.output_dir,
	K=FLAGS.K,
	J=FLAGS.J,
	F=FLAGS.F,
	eps=FLAGS.eps,
	ode_int_method=FLAGS.ode_int_method,
	ode_int_atol=FLAGS.ode_int_atol,
	ode_int_rtol=FLAGS.ode_int_rtol,
	ode_int_max_step=FLAGS.ode_int_max_step,
	delta_t = FLAGS.delta_t,
	t_synch = FLAGS.t_synch,
	t_train = FLAGS.t_train,
	t_invariant_measure = FLAGS.t_invariant_measure,
	rng_seed = FLAGS.rng_seed):

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
	if rng_seed:
		np.random.seed(rng_seed)

	t_eval = np.arange(0, (t_synch+t_invariant_measure), delta_t)
	y0 = ODE.get_inits().squeeze()
	t0 = time()
	sol = solve_ivp(fun=lambda t, y: ODE.rhs(y, t), t_span=(t_eval[0], t_eval[-1]), y0=y0, method=ode_int_method, rtol=ode_int_rtol, atol=ode_int_atol, max_step=ode_int_max_step, t_eval=t_eval)
	y_clean = sol.y.T
	print('Generated invariant measure Testing data in:', (time()-t0)/60,'minutes')

	# Remove first 500 and plot KDE
	X_test = y_clean[ntsynch:,:K]
	np.savez(testing_fname, X_test=X_test, ntsynch=ntsynch, t_eval=t_eval, y0=y0, K=K)

	return

def plot_data(testing_fname=os.path.join(FLAGS.output_dir, FLAGS.testing_fname),
	training_fname=os.path.join(FLAGS.output_dir,FLAGS.training_fname),
	dima_data_path=FLAGS.dima_data_path,
	output_dir = FLAGS.output_dir,
	n_subsample_gp=FLAGS.n_subsample_gp,
	n_subsample_kde=FLAGS.n_subsample_kde,
	n_restarts_optimizer=FLAGS.n_restarts_optimizer,
	K=FLAGS.K,
	J=FLAGS.J,
	F=FLAGS.F,
	eps=FLAGS.eps,
	ode_int_method=FLAGS.ode_int_method,
	ode_int_atol=FLAGS.ode_int_atol,
	ode_int_rtol=FLAGS.ode_int_rtol,
	ode_int_max_step=FLAGS.ode_int_max_step,
	alpha_list_cont = [1e-10],
	alpha_list_discrete = [1e-10],
	fit_dima=False,
	delta_t=FLAGS.delta_t,
	T_plot=10):

	# plot inds for trajectories
	t_plot = np.arange(0, T_plot, delta_t)
	n_plot = len(t_plot)

	# output dir
	output_fname = os.path.join(output_dir,'eliminate_dima.png')

	# initialize ode object
	ODE = L96M(K=K, J=J, F=F, eps=eps, dima_style=True)
	ODE.set_stencil() # this is a default, empty usage that is required
	state_limits = ODE.get_state_limits()

	# get initial colors
	prop_cycle = plt.rcParams['axes.prop_cycle']
	color_list = prop_cycle.by_key()['color']

	foo = np.load(training_fname)
	goo = np.load(testing_fname)

	# Read in data
	X = foo['X_train'][:-1,:].reshape(-1,1)
	Y_true = foo['Ybar_true'].reshape(-1,1)
	Y_inferred = foo['Ybar_data_inferred'].reshape(-1,1)
	train_inds = get_inds(N_total=X.shape[0], N_subsample=n_subsample_gp)
	X_pred = np.arange(np.min(X),np.max(X),0.01).reshape(-1, 1) # mesh to evaluate GPR

	X_test = goo['X_test'].reshape(-1,1)
	t_eval = goo['t_eval']
	ntsynch = goo['ntsynch']
	K = goo['K']
	# y0 = goo['y0'][:K]
	y0 = goo['X_test'][0,:]
	test_inds = get_inds(N_total=X_test.shape[0], N_subsample=n_subsample_kde)
	fig, (ax_gp, ax_kde) = plt.subplots(1,2,figsize=[9,5])
	fig_discrete, (ax_gp_discrete, ax_kde_discrete) = plt.subplots(1,2, figsize=[9,5])
	t0 = time()
	sns.kdeplot(X_test[test_inds].squeeze(), ax=ax_kde, label='RHS = Full Multiscale', color='black', linestyle='-')
	sns.kdeplot(X_test[test_inds].squeeze(), ax=ax_kde_discrete, label='RHS = Full Multiscale', color='black', linestyle='-')
	ax_gp.legend(loc='best', prop={'size': 5})
	ax_kde.legend(loc='lower center', prop={'size': 4})
	fig.savefig(fname=output_fname, dpi=300)
	print('Plotted matt-KDE of invariant measure in:', (time()-t0)/60,'minutes')

	if fit_dima:
		# plot Dimas data invariant measure
		X_dima = np.load(dima_data_path)
		# sns.kdeplot(X_dima[:,0].squeeze(), ax=ax_kde, label='Dima-True', color='black', linestyle=':')
		# print('Plotted dima-KDE of invariant measure in:', (time()-t0)/60,'minutes')
		dima_inds = get_inds(N_total=X_dima.shape[0], N_subsample=n_subsample_gp)

	ax_kde.legend()

	# plot training data
	ax_gp.plot(X, np.mean(ODE.hx)*Y_true, 'o', markersize=2, color='gray', alpha=0.8, label='True Training Data (all)')
	ax_gp.plot(X[train_inds], np.mean(ODE.hx)*Y_true[train_inds], 'o', markersize=2, color='red', alpha=0.8, label='True Training Data (subset={n_subsample_gp})'.format(n_subsample_gp=n_subsample_gp))
	ax_gp.plot(X[train_inds], np.mean(ODE.hx)*Y_inferred[train_inds], '+', linewidth=1, markersize=3, markeredgewidth=1, color='green', alpha=0.8, label='Approximate Training Data (subset={n_subsample_gp})'.format(n_subsample_gp=n_subsample_gp))
	ax_gp.legend(loc='best', prop={'size': 5})
	ax_kde.legend(loc='lower center', prop={'size': 4})
	fig.savefig(fname=output_fname, dpi=300)

	def plot_traj(X_learned, plot_fname, t_plot=t_plot, X_true=goo['X_test'][:n_plot,:K], state_limits=state_limits):
		K = X_true.shape[1]
		fig_traj, (ax_test) = plt.subplots(K,1,figsize=[8,10])
		for k in range(K):
			ax_test[k].plot(t_plot, X_true[:,k], color='black', label='True')
			ax_test[k].plot(t_plot, X_learned[:,k], color='blue', label='Slow + __')
			ax_test[k].set_ylabel(r'$X_{k}$'.format(k=k))
			ax_test[k].set_ylim(state_limits)
		ax_test[-1].set_xlabel('time')
		ax_test[0].legend(loc='center right')
		fig_traj.suptitle('Testing Trajectories')
		fig_traj.savefig(fname=plot_fname, dip=300)
		plt.close(fig_traj)
		return

	# check B0 predictor (hy*x)
	ODE.set_G0_predictor()
	# g0_mean = ODE.hy*X_pred
	# ax_gp.plot(X_pred, np.mean(ODE.hx)*g0_mean, color='black', linestyle='--', label='G0')
	t0 = time()
	sol = solve_ivp(fun=lambda t, y: ODE.regressed(y, t), t_span=(t_eval[0], t_eval[-1]), y0=y0, method=ode_int_method, rtol=ode_int_rtol, atol=ode_int_atol, max_step=ode_int_max_step, t_eval=t_eval)
	y_clean = sol.y.T
	X_test_G0 = y_clean[ntsynch:,:K].reshape(-1, 1)
	print('Generated invariant measure for G0:', (time()-t0)/60,'minutes')
	# sns.kdeplot(X_test_G0[test_inds].squeeze(), ax=ax_kde, color='gray', linestyle='-', label='G0')
	# plot trajectory fits
	plot_traj(X_learned=y_clean[:n_plot,:K], plot_fname=os.path.join(output_dir,'trajectory_slow_plus_cX.png'))

	# check null predictor (0)
	ODE.set_null_predictor()
	# g0_mean = ODE.hy*X_pred
	# ax_gp.plot(X_pred, np.mean(ODE.hx)*g0_mean, color='black', linestyle='--', label='G0')
	t0 = time()
	sol = solve_ivp(fun=lambda t, y: ODE.regressed(y, t), t_span=(t_eval[0], t_eval[-1]), y0=y0, method=ode_int_method, rtol=ode_int_rtol, atol=ode_int_atol, max_step=ode_int_max_step, t_eval=t_eval)
	y_clean = sol.y.T
	X_test_null = y_clean[ntsynch:,:K].reshape(-1, 1)
	print('Generated invariant measure for G0:', (time()-t0)/60,'minutes')
	sns.kdeplot(X_test_null[test_inds].squeeze(), ax=ax_kde, color='gray', linestyle='-', label='RHS = Slow')
	sns.kdeplot(X_test_null[test_inds].squeeze(), ax=ax_kde_discrete, color='gray', linestyle='-', label='RHS = Slow')
	ax_gp.legend(loc='best', prop={'size': 5})
	ax_kde.legend(loc='lower center', prop={'size': 4})
	fig.savefig(fname=output_fname, dpi=300)
	# plot trajectory fits
	plot_traj(X_learned=y_clean[:n_plot,:K], plot_fname=os.path.join(output_dir,'trajectory_slow_plus_zero.png'))

	# first, make training data for discrete GP training
	# GP(X_j) ~= Xtrue_{j+1} - Psi_slow(Xtrue_j), where Xtrue are true solutions of the slow variable
	X_train_gp = foo['X_train']
	slow_preds = np.zeros((X_train_gp.shape[0]-1, X_train_gp.shape[1]))
	for j in range(X_train_gp.shape[0]-1):
		ic = X_train_gp[j,:] # initial condition
		sol = solve_ivp(fun=lambda t, y: ODE.slow(y, t), t_span=(0, delta_t), y0=ic, method=ode_int_method, rtol=ode_int_rtol, atol=ode_int_atol, max_step=ode_int_max_step, t_eval=np.array([delta_t]))
		slow_preds[j,:] = sol.y.T # prediction Psi_slow(Xtrue_j)
	y_train_gp = X_train_gp[1:,:] - slow_preds # get the residuals
	X_train_gp = X_train_gp[:-1,:] # get the inputs
	gp_train_inds_full = get_inds(N_total=X_train_gp.shape[0], N_subsample=n_subsample_gp)
	gp_train_inds_share = get_inds(N_total=X_train_gp.reshape(-1,1).shape[0], N_subsample=n_subsample_gp)

	ax_kde_discrete.set_xlabel(r'$X_k$')
	ax_gp_discrete.plot(X_train_gp.reshape(-1,1), y_train_gp.reshape(-1,1)/delta_t, 'o', markersize=2, color='gray', alpha=0.8, label='Training Data (all)')
	ax_gp_discrete.plot(X_train_gp.reshape(-1,1)[gp_train_inds_share], y_train_gp.reshape(-1,1)[gp_train_inds_share]/delta_t, 'o', markersize=2, color='red', alpha=0.8, label='GP-share Data (subset={n_subsample_gp})'.format(n_subsample_gp=n_subsample_gp))
	ax_gp_discrete.plot(X_train_gp[gp_train_inds_full,:].reshape(-1,1), y_train_gp[gp_train_inds_full,:].reshape(-1,1)/delta_t, '+', markersize=3, color='cyan', alpha=0.8, label='GP-full Data (subset={n_subsample_gp})'.format(n_subsample_gp=n_subsample_gp))
	fig_discrete.suptitle('GPR fits to errors of discrete slow-only forward-map')
	ax_gp_discrete.set_xlabel(r'$X^{(t)}_k$')
	ax_gp_discrete.set_ylabel(r'$[X^{(t+1)}_k - \Psi_0(X^{(t)})_k] / \Delta t$')
	ax_gp_discrete.legend(loc='best', prop={'size': 5})
	fig_discrete.savefig(fname=os.path.join(output_dir,'gp_discrete_fit.png'), dpi=300)
	for c in range(len(alpha_list_discrete)):
		alpha = alpha_list_discrete[c]
		color = color_list[c]
		print('alpha=',alpha, color)

		# intialize gpr
		GP_ker = ConstantKernel(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-10, 1e+6)) + WhiteKernel(1.0, (1e-10, 1e6))
		my_gpr = GaussianProcessRegressor(
			kernel = GP_ker,
			n_restarts_optimizer = n_restarts_optimizer,
			alpha = alpha
		)

		######### GP-fulltofull #######
		# fit GP to residuals of discrete operator
		gpr_discrete_full = my_gpr.fit(X=X_train_gp[gp_train_inds_full,:], y=y_train_gp[gp_train_inds_full,:]/delta_t)
		X_pred_outer = np.outer(X_pred,np.ones(K))
		gpr_discrete_full_mean = gpr_discrete_full.predict(X_pred_outer, return_std=False) # evaluate at [0,0,0,0], [0.01,0.01,0.01,0.01], etc.
		# plot training data
		ax_gp_discrete.plot(X_pred_outer.reshape(-1,1), gpr_discrete_full_mean.reshape(-1,1), color=color, linestyle='', marker='+', markeredgewidth=0.1, markersize=3, label=r'$\Phi_{{\theta}}(X^{{(t)}}_k)$ ({kernel})'.format(kernel=my_gpr.kernel_))
		ax_gp_discrete.legend(loc='best', prop={'size': 5})
		fig_discrete.savefig(fname=os.path.join(output_dir,'gp_discrete_fit.png'), dpi=300)
		t0 = time()
		# now generate a test trajectory using the learned GPR
		y_clean = np.zeros((len(t_eval), K))
		y_clean[0,:] = y0
		for j in range(len(t_eval)-1):
			ic_discrete = y_clean[j,:]
			sol = solve_ivp(fun=lambda t, y: ODE.slow(y, t), t_span=(0, delta_t), y0=ic_discrete, method=ode_int_method, rtol=ode_int_rtol, atol=ode_int_atol, max_step=ode_int_max_step, t_eval=[delta_t])
			y_pred = sol.y.squeeze()
			# compute fulltofull GPR correction
			y_pred += delta_t*gpr_discrete_full.predict(ic_discrete.reshape(1,-1), return_std=False).squeeze()
			y_clean[j+1,:] = y_pred
		X_test_gpr_discrete_full = y_clean[ntsynch:,:K].reshape(-1, 1)
		print('Generated invariant measure for GP-discrete-full:', (time()-t0)/60,'minutes')
		plot_traj(X_learned=y_clean[:n_plot,:K], plot_fname=os.path.join(output_dir,'trajectory_discrete_fullGP_alpha{alpha}.png'.format(alpha=alpha)))
		sns.kdeplot(X_test_gpr_discrete_full[test_inds].squeeze(), ax=ax_kde_discrete, color=color, linestyle='', marker='o', markeredgewidth=1, markersize=2, label=r'$X_{{k+1}} = \Psi_0(X_k) + \Phi_{{\theta}}(X_k)$ ({kernel})'.format(kernel=my_gpr.kernel_))
		ax_gp_discrete.legend(loc='best', prop={'size': 5})
		ax_kde_discrete.legend(loc='lower center', prop={'size': 4})
		fig_discrete.savefig(fname=os.path.join(output_dir,'gp_discrete_fit.png'), dpi=300)


		######### GP-share #######
		# fit GP to residuals of discrete operator
		gpr_discrete_share = my_gpr.fit(X=X_train_gp.reshape(-1,1)[gp_train_inds_share], y=y_train_gp.reshape(-1,1)[gp_train_inds_share]/delta_t)
		gpr_discrete_share_mean = gpr_discrete_share.predict(X_pred, return_std=False)
		# plot training data
		ax_gp_discrete.plot(X_pred, gpr_discrete_share_mean, color=color, linestyle='-', label=r'$\bar{{\Phi}}_{{\theta}}(X^{{(t)}}_k)$ ({kernel})'.format(kernel=my_gpr.kernel_))
		ax_gp_discrete.legend(loc='best', prop={'size': 5})
		fig_discrete.savefig(fname=os.path.join(output_dir,'gp_discrete_fit.png'), dpi=300)
		t0 = time()
		# now generate a test trajectory using the learned GPR
		y_clean = np.zeros((len(t_eval), K))
		y_clean[0,:] = y0
		for j in range(len(t_eval)-1):
			ic_discrete = y_clean[j,:]
			sol = solve_ivp(fun=lambda t, y: ODE.slow(y, t), t_span=(0, delta_t), y0=ic_discrete, method=ode_int_method, rtol=ode_int_rtol, atol=ode_int_atol, max_step=ode_int_max_step, t_eval=[delta_t])
			y_pred = sol.y.squeeze()
			# compute shared GPR correction
			for k in range(K):
				y_pred[k] += delta_t*gpr_discrete_share.predict(ic_discrete[k].reshape(1,-1), return_std=False)
			y_clean[j+1,:] = y_pred
		X_test_gpr_discrete_share = y_clean[ntsynch:,:K].reshape(-1, 1)
		print('Generated invariant measure for GP-discrete-share:', (time()-t0)/60,'minutes')
		plot_traj(X_learned=y_clean[:n_plot,:K], plot_fname=os.path.join(output_dir,'trajectory_discrete_shareGP_alpha{alpha}.png'.format(alpha=alpha)))
		sns.kdeplot(X_test_gpr_discrete_share[test_inds].squeeze(), ax=ax_kde_discrete, color=color, linestyle='-', label=r'$X_{{k+1}} = \Psi_0(X_k) + \bar{{\Phi}}_{{\theta}}(X_k)$ ({kernel})'.format(kernel=my_gpr.kernel_))
		ax_kde_discrete.legend(loc='lower center', prop={'size': 4})
		fig_discrete.savefig(fname=os.path.join(output_dir,'gp_discrete_fit.png'), dpi=300)

		np.savez(os.path.join(output_dir,'test_output_discrete_{alpha}.npz'.format(alpha=alpha)),
		X_test_gpr_discrete_share=X_test_gpr_discrete_share,
		X_test_gpr_discrete_full=X_test_gpr_discrete_full,
		X_test=X_test,
		X_test_null=X_test_null)
	plt.close(fig_discrete)


	# now run continuous RHS learning
	ax_kde.set_xlabel(r'$X_k$')
	ax_kde.legend(loc='lower center', prop={'size': 4})
	ax_gp.set_xlabel(r'$X_k$')
	ax_gp.set_ylabel(r'$h_x \bar{Y}_k$')
	ax_gp.legend(loc='best', prop={'size': 5})
	fig.savefig(fname=output_fname, dpi=300)
	for c in range(len(alpha_list_cont)):
		alpha = alpha_list_cont[c]
		color = color_list[c]
		print('alpha=',alpha, color)

		# intialize gpr
		GP_ker = ConstantKernel(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-10, 1e+6)) + WhiteKernel(1.0, (1e-10, 1e6))
		my_gpr = GaussianProcessRegressor(
			kernel = GP_ker,
			n_restarts_optimizer = n_restarts_optimizer,
			alpha = alpha
		)

		# fit GPR-Ybartrue to Xk vs Ybar-true
		gpr_true = my_gpr.fit(X=X[train_inds], y=np.mean(ODE.hx)*Y_true[train_inds])
		gpr_true_mean = gpr_true.predict(X_pred, return_std=False)
		# now run gp-corrected ODE
		ODE.set_predictor(gpr_true.predict)
		t0 = time()
		sol = solve_ivp(fun=lambda t, y: ODE.regressed(y, t), t_span=(t_eval[0], t_eval[-1]), y0=y0, method=ode_int_method, rtol=ode_int_rtol, atol=ode_int_atol, max_step=ode_int_max_step, t_eval=t_eval)
		y_clean = sol.y.T
		X_test_gpr_true = y_clean[ntsynch:,:K].reshape(-1, 1)
		print('Generated invariant measure for GP-true:', (time()-t0)/60,'minutes')
		plot_traj(X_learned=y_clean[:n_plot,:K], plot_fname=os.path.join(output_dir,'trajectory_YbarTrue_alpha{alpha}.png'.format(alpha=alpha)))
		sns.kdeplot(X_test_gpr_true[test_inds].squeeze(), ax=ax_kde, color=color, linestyle='-', label='RHS = Slow + GP (True Y-avg) ({kernel})'.format(kernel=my_gpr.kernel_))
		ax_gp.plot(X_pred, gpr_true_mean, color=color, linestyle='-', label='GP (True Y-avg) ({kernel})'.format(kernel=my_gpr.kernel_))
		ax_gp.legend(loc='best', prop={'size': 5})
		ax_kde.legend(loc='lower center', prop={'size': 4})
		fig.savefig(fname=output_fname, dpi=300)

		# fit GPR-Ybarpprox to Xk vs Ybar-infer
		gpr_approx = my_gpr.fit(X=X[train_inds], y=np.mean(ODE.hx)*Y_inferred[train_inds])
		gpr_approx_mean = gpr_approx.predict(X_pred, return_std=False)
		# now run gp-corrected ODE
		ODE.set_predictor(gpr_approx.predict)
		t0 = time()
		sol = solve_ivp(fun=lambda t, y: ODE.regressed(y, t), t_span=(t_eval[0], t_eval[-1]), y0=y0, method=ode_int_method, rtol=ode_int_rtol, atol=ode_int_atol, max_step=ode_int_max_step, t_eval=t_eval)
		y_clean = sol.y.T
		X_test_gpr_approx = y_clean[ntsynch:,:K].reshape(-1, 1)
		print('Generated invariant measure for GP-approx:', (time()-t0)/60,'minutes')
		plot_traj(X_learned=y_clean[:n_plot,:K], plot_fname=os.path.join(output_dir,'trajectory_YbarInfer_alpha{alpha}.png'.format(alpha=alpha)))
		sns.kdeplot(X_test_gpr_approx[test_inds].squeeze(), ax=ax_kde, color=color, linestyle='--', label='RHS = Slow + GP (Approx Y-avg) ({kernel})'.format(kernel=my_gpr.kernel_))
		ax_gp.plot(X_pred, gpr_approx_mean, color=color, linestyle='--', label='GP (Inferred Y-avg) ({kernel})'.format(kernel=my_gpr.kernel_))
		ax_gp.legend(loc='best', prop={'size': 5})
		ax_kde.legend(loc='lower center', prop={'size': 4})
		fig.savefig(fname=output_fname, dpi=300)

		# save figure after each loop
		np.savez(os.path.join(output_dir,'test_output_continuous_{alpha}.npz'.format(alpha=alpha)),
				X_test_gpr_true=X_test_gpr_true,
				X_test_gpr_approx=X_test_gpr_approx,
				X_test=X_test,
				X_test_null=X_test_null)

	# dont be a slob...close the fig when you're done!
	plt.close(fig)
	return


if __name__ == '__main__':
	eliminate_dima()
	plot_data()


