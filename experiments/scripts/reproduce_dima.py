import os
import json
import numpy as np
import argparse
from time import time
from utils import phase_plot, kl4dummies, fname_append
from check_L96_chaos import make_traj_plots
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from odelibrary import L96M
from integratorlibrary import get_custom_solver
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from statsmodels.tsa.stattools import acf
import pdb

#Note: to use custom euler integrator, simply set delta_t
# and let --ode_int_method='Euler'

def get_inds(N_total, N_subsample):
	if N_total > N_subsample:
		my_inds = np.random.choice(np.arange(N_total), N_subsample, replace=False)
	else:
		my_inds = np.arange(N_total)
	return my_inds

def make_data(
	testing_fname,
	training_fname,
	output_dir,
	K,
	J,
	F,
	eps,
	hx,
	ode_int_method,
	ode_int_atol,
	ode_int_rtol,
	ode_int_max_step,
	delta_t,
	t_synch,
	t_train,
	t_invariant_measure,
	rng_seed,
	n_test_traj,
	t_test_traj,
	**kwargs):

	os.makedirs(output_dir, exist_ok=True)

	# First, check settings for chaos
	sim_model_params = {'ode_params': (),
					'ode_int_method':'RK45',
					'ode_int_atol':1e-6,
					'ode_int_rtol':1e-3,
					'ode_int_max_step':1e-3}
	make_traj_plots(n_inits=4, sd_perturb=0.01, sim_model_params=sim_model_params, output_dir=output_dir, K=K, J=J, F=F, eps=eps, hx=hx, delta_t=delta_t, T=15, decoupled=False)


	# initialize ode object
	ODE = L96M(K=K, J=J, F=F, eps=eps, hx=hx)
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

	np.savez(training_fname, X_train=X_train, y_fast=y_fast, Ybar_true=Ybar_true, Ybar_data_inferred=Ybar_data_inferred)

	phase_plot(data=X_train, output_fname=os.path.join(output_dir,'phase_plot_training_data_SLOW.png'), delta_t=delta_t, wspace=0.35, hspace=0.35)
	phase_plot(data=X_train, output_fname=os.path.join(output_dir,'phase_plot_training_data_SLOW.png'), delta_t=delta_t, wspace=0.35, hspace=0.35, mode='density')
	phase_plot(data=y_fast[:,:J], state_names=[r'$Y_{{{ind},1}}$'.format(ind=j+1) for j in range(J)], output_fname=os.path.join(output_dir,'phase_plot_training_data_FAST.png'), delta_t=delta_t, wspace=0.35, hspace=0.35)
	phase_plot(data=y_fast[:,:J], state_names=[r'$Y_{{{ind},1}}$'.format(ind=j+1) for j in range(J)], output_fname=os.path.join(output_dir,'phase_plot_training_data_FAST.png'), delta_t=delta_t, wspace=0.35, hspace=0.35, mode='density')

	all_kdes_plot(data=X_train, output_fname=os.path.join(output_dir,'all_kdes_SLOW.png'))
	all_kdes_plot(data=y_fast[:,:J], output_fname=os.path.join(output_dir,'all_kdes_FAST.png'))

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

	# create short testing trajectories with initial conditions sampled from invariant density
	t_eval_traj = np.arange(0, t_test_traj, delta_t)
	ic_inds = get_inds(N_total=X_test.shape[0], N_subsample=n_test_traj)
	X_test_traj = np.zeros((n_test_traj,len(t_eval_traj),K))
	for c in range(n_test_traj):
		ic = y_clean[ntsynch+ic_inds[c],:]
		t0 = time()
		sol = solve_ivp(fun=lambda t, y: ODE.rhs(y, t), t_span=(t_eval[0], t_eval[-1]), y0=ic, method=ode_int_method, rtol=ode_int_rtol, atol=ode_int_atol, max_step=ode_int_max_step, t_eval=t_eval_traj)
		X_test_traj[c,:,:] = sol.y.T[:,:K]

	np.savez(testing_fname, X_test_traj=X_test_traj, t_eval_traj=t_eval_traj, X_test=X_test, ntsynch=ntsynch, t_eval=t_eval, y0=y0, K=K)


	return


def traj_div_time(Xtrue, Xpred, delta_t, avg_output, thresh):
	# avg_output = np.mean(Xtrue**2)**0.5
	pw_loss = np.zeros((Xtrue.shape[0]))
	for j in range(Xtrue.shape[0]):
		pw_loss[j] = sum((Xtrue[j,:] - Xpred[j,:])**2)**0.5 / avg_output
	t_valid = delta_t*np.argmax(pw_loss > thresh)
	return t_valid

def run_traintest(testing_fname,
	training_fname,
	master_output_fname,
	output_dir,
	n_subsample_gp,
	n_subsample_kde,
	n_restarts_optimizer,
	K,
	J,
	F,
	eps,
	hx,
	psi0_ode_int_method,
	psi0_ode_int_atol,
	psi0_ode_int_rtol,
	psi0_ode_int_max_step,
	testcontinuous_ode_int_method,
	testcontinuous_ode_int_atol,
	testcontinuous_ode_int_rtol,
	testcontinuous_ode_int_max_step,
	delta_t,
	T_plot=10,
	T_acf=10,
	alpha = 1e-10,
	t_valid_thresh=0.4,
	**kwargs):

	try:
		foo = np.load(training_fname)
		goo = np.load(testing_fname)
	except:
		print('Unable to load training data---no plots were made!')
		return

	n_subsample_gp = int(n_subsample_gp)
	n_subsample_kde = int(n_subsample_kde)
	n_restarts_optimizer = int(n_restarts_optimizer)

	# make output_dir
	os.makedirs(output_dir, exist_ok=True)

	# set up master output dictionary to be saved to file
	try:
		boomaster = np.load(master_output_fname, allow_pickle=True)
		# convert npzFileObject to a dictionary
		# https://stackoverflow.com/questions/32682928/loading-arrays-from-numpy-npz-files-in-python
		master_output_dict = dict(zip(("{}".format(k) for k in boomaster), (boomaster[k] for k in boomaster)))
	except:
		master_output_dict = {}

	# set up test-traj dict
	name_list = ['RHS = Slow',
				'Discrete Full',
				'Discrete Share',
				'Continuous Share Y-True',
				'Continuous Share Y-Approx']

	# set up acf lags
	nlags = int(T_acf/delta_t) - 1
	t_acf_plot = np.arange(0, T_acf, delta_t)

	# plot inds for trajectories
	t_plot = np.arange(0, T_plot, delta_t)
	n_plot = len(t_plot)

	# output dir
	output_fname = os.path.join(output_dir,'continuous_fits.png')

	# initialize ode object
	ODE = L96M(K=K, J=J, F=F, eps=eps, hx=hx, dima_style=False)

	# First, check settings for chaos
	sim_model_params = {'ode_params': (),
					'ode_int_method': testcontinuous_ode_int_method,
					'ode_int_atol': testcontinuous_ode_int_atol,
					'ode_int_rtol': testcontinuous_ode_int_rtol,
					'ode_int_max_step': testcontinuous_ode_int_max_step}
	make_traj_plots(n_inits=4, sd_perturb=0.01, sim_model_params=sim_model_params, output_dir=output_dir, K=K, J=J, F=F, eps=eps, hx=hx, delta_t=delta_t, T=15, decoupled=False)



	ODE.set_stencil() # this is a default, empty usage that is required
	state_limits = ODE.get_state_limits()




	# get initial colors
	prop_cycle = plt.rcParams['axes.prop_cycle']
	color_list = prop_cycle.by_key()['color']

	# plot a phase plot of the training data
	phase_plot(data=foo['X_train'], output_fname=os.path.join(output_dir,'phase_plot_training_data_SLOW.png'), delta_t=delta_t, wspace=0.35, hspace=0.35)
	phase_plot(data=foo['X_train'], output_fname=os.path.join(output_dir,'phase_plot_training_data_SLOW.png'), delta_t=delta_t, wspace=0.35, hspace=0.35, mode='density')
	all_kdes_plot(data=foo['X_train'], output_fname=os.path.join(output_dir,'all_kdes_SLOW.png'))
	try:
		phase_plot(data=foo['y_fast'][:,:J], state_names=[r'$Y_{{{ind},1}}$'.format(ind=j+1) for j in range(J)], output_fname=os.path.join(output_dir,'phase_plot_training_data_FAST.png'), delta_t=delta_t, wspace=0.35, hspace=0.35)
		phase_plot(data=foo['y_fast'][:,:J], state_names=[r'$Y_{{{ind},1}}$'.format(ind=j+1) for j in range(J)], output_fname=os.path.join(output_dir,'phase_plot_training_data_FAST.png'), delta_t=delta_t, wspace=0.35, hspace=0.35, mode='density')
		all_kdes_plot(data=foo['y_fast'][:,:J], output_fname=os.path.join(output_dir,'all_kdes_FAST.png'))
	except:
		print('Could not plot fast data...it wasnt saved in old data-generation runs')

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
	fig, (ax_gp, ax_kde, ax_acf, ax_tvalid) = plt.subplots(1,4,figsize=[24,7])
	fig_discrete, (ax_gp_discrete, ax_kde_discrete, ax_acf_discrete, ax_tvalid_discrete) = plt.subplots(1,4, figsize=[24,7])
	t0 = time()
	sns.kdeplot(X_test[test_inds].squeeze(), ax=ax_kde, label='RHS = Full Multiscale', color='black', linestyle='-')
	sns.kdeplot(X_test[test_inds].squeeze(), ax=ax_kde_discrete, label='RHS = Full Multiscale', color='black', linestyle='-')
	ax_gp.legend(loc='best', prop={'size': 5.5})
	ax_acf.legend(loc='best', prop={'size': 8})
	ax_kde.legend(loc='lower center', prop={'size': 8})
	fig.savefig(fname=output_fname, dpi=300)
	print('Plotted matt-KDE of invariant measure in:', (time()-t0)/60,'minutes')

	# read in short test trajectories
	X_test_traj = goo['X_test_traj']
	n_test_traj = X_test_traj.shape[0]
	t_valid = {nm: np.nan*np.ones(n_test_traj) for nm in name_list}

	# plot training data
	ax_gp.plot(X, np.mean(ODE.hx)*Y_true, 'o', markersize=2, color='gray', alpha=0.8, label='True Training Data (all)')
	ax_gp.plot(X[train_inds], np.mean(ODE.hx)*Y_true[train_inds], 'o', markersize=2, color='red', alpha=0.8, label='True Training Data (subset={n_subsample_gp})'.format(n_subsample_gp=n_subsample_gp))
	ax_gp.plot(X[train_inds], np.mean(ODE.hx)*Y_inferred[train_inds], '+', linewidth=1, markersize=3, markeredgewidth=1, color='green', alpha=0.8, label='Approximate Training Data (subset={n_subsample_gp})'.format(n_subsample_gp=n_subsample_gp))
	ax_gp.legend(loc='best', prop={'size': 5.5})
	ax_acf.legend(loc='best', prop={'size': 8})
	ax_kde.legend(loc='lower center', prop={'size': 8})

	T_test_acf = acf(goo['X_test'][:,0], fft=True, nlags=nlags) #look at first component
	ax_acf_discrete.plot(t_acf_plot, T_test_acf, color='black', label='RHS = Full Multiscale')
	ax_acf.plot(t_acf_plot, T_test_acf, color='black', label='RHS = Full Multiscale')
	ax_acf.set_ylabel(r'Autocorrelation($X_0$)')
	ax_acf.set_xlabel('Time')
	ax_acf.set_xscale('log')
	ax_acf_discrete.set_ylabel(r'Autocorrelation($X_0$)')
	ax_acf_discrete.set_xlabel('Time')
	ax_acf_discrete.set_xscale('log')
	fig.savefig(fname=output_fname, dpi=300)
	fig_discrete.savefig(fname=os.path.join(output_dir,'gp_discrete_fits.png'), dpi=300)

	# compute time-avg-norm
	avg_output = np.mean(goo['X_test']**2)**0.5


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
	# ODE.set_G0_predictor()
	# ODE.dima_style = True
	# g0_mean = ODE.hy*X_pred
	# ax_gp.plot(X_pred, np.mean(ODE.hx)*g0_mean, color='black', linestyle='--', label='G0')
	# t0 = time()
	# sol = solve_ivp(fun=lambda t, y: ODE.regressed(y, t), t_span=(t_eval[0], t_eval[-1]), y0=y0, method=testcontinuous_ode_int_method, rtol=testcontinuous_ode_int_rtol, atol=testcontinuous_ode_int_atol, max_step=testcontinuous_ode_int_max_step, t_eval=t_eval)
	# y_clean = sol.y.T
	# X_test_G0 = y_clean[ntsynch:,:K].reshape(-1, 1)
	# print('Generated invariant measure for G0:', (time()-t0)/60,'minutes')
	# sns.kdeplot(X_test_G0[test_inds].squeeze(), ax=ax_kde, color='gray', linestyle='-', label='G0')
	# plot trajectory fits
	# plot_traj(X_learned=y_clean[:n_plot,:K], plot_fname=os.path.join(output_dir,'trajectory_slow_plus_cX.png'))


	# check null predictor (0)
	foo_nm = 'null'
	ODE.set_null_predictor()
	ODE.dima_style = False
	# g0_mean = ODE.hy*X_pred
	# ax_gp.plot(X_pred, np.mean(ODE.hx)*g0_mean, color='black', linestyle='--', label='G0')
	t0 = time()
	try:
		y_clean = master_output_dict[foo_nm+'_y_clean']
	except:
		sol = solve_ivp(fun=lambda t, y: ODE.regressed(y, t), t_span=(t_eval[0], t_eval[-1]), y0=y0, method=testcontinuous_ode_int_method, rtol=testcontinuous_ode_int_rtol, atol=testcontinuous_ode_int_atol, max_step=testcontinuous_ode_int_max_step, t_eval=t_eval)
		y_clean = sol.y.T
		master_output_dict[foo_nm+'_y_clean'] = y_clean
		np.savez(master_output_fname,**master_output_dict)
	X_test_null = y_clean[ntsynch:,:K].reshape(-1, 1)
	T_test_null_acf = acf(y_clean[ntsynch:,0], fft=True, nlags=nlags) #look at first component
	print('Generated invariant measure for RHS=Slow + 0:', (time()-t0)/60,'minutes')
	sns.kdeplot(X_test_null[test_inds].squeeze(), ax=ax_kde, color='gray', linestyle='-', label='RHS = Slow')
	sns.kdeplot(X_test_null[test_inds].squeeze(), ax=ax_kde_discrete, color='gray', linestyle='-', label=r'$X_{k+1} = \Psi_0(X_k)$')
	ax_acf.plot(t_acf_plot, T_test_null_acf, color='gray', label='RHS = Slow')
	ax_acf_discrete.plot(t_acf_plot, T_test_null_acf, color='gray', label=r'$X_{k+1} = \Psi_0(X_k)$')
	ax_gp.legend(loc='best', prop={'size': 5.5})
	ax_acf.legend(loc='best', prop={'size': 8})
	ax_kde.legend(loc='lower center', prop={'size': 8})
	fig.savefig(fname=output_fname, dpi=300)
	# plot trajectory fits
	plot_traj(X_learned=y_clean[:n_plot,:K], plot_fname=os.path.join(output_dir,'trajectory_slow_plus_zero.png'))
	# run model against test trajectories

	try:
		test_traj_preds = master_output_dict[foo_nm+'_test_traj_preds']
		do_compute = False
	except:
		test_traj_preds = np.zeros(X_test_traj.shape)
		do_compute = True
	for t in range(n_test_traj):
		if do_compute:
			ic = X_test_traj[t,0,:]
			for j in range(X_test_traj.shape[1]):
				test_traj_preds[t,j,:] = ic # prediction Psi_slow(Xtrue_j)
				sol = solve_ivp(fun=lambda t, y: ODE.regressed(y, t), t_span=(0, delta_t), y0=ic, method=testcontinuous_ode_int_method, rtol=testcontinuous_ode_int_rtol, atol=testcontinuous_ode_int_atol, max_step=testcontinuous_ode_int_max_step, t_eval=np.array([delta_t]))
				ic = sol.y.T.squeeze()
			master_output_dict[foo_nm+'_test_traj_preds'] = test_traj_preds
			np.savez(master_output_fname,**master_output_dict)

		# compute traj error
		tval_foo = traj_div_time(Xtrue=X_test_traj[t,:,:], Xpred=test_traj_preds[t,:,:], delta_t=delta_t, avg_output=avg_output, thresh=t_valid_thresh)
		t_valid['RHS = Slow'][t] = tval_foo

	for ax in [ax_tvalid_discrete, ax_tvalid]:
		sns.boxplot(ax=ax, data=pd.DataFrame(t_valid), color='orchid')
		ax.set_xticklabels(ax.get_xticklabels(), rotation=20, horizontalalignment='right', fontsize='small')
		ax.set_ylabel('Validity Time')
	fig_discrete.savefig(fname=os.path.join(output_dir,'gp_discrete_fits.png'), dpi=300)
	fig.savefig(fname=os.path.join(output_dir,'continuous_fits.png'), dpi=300)

	# first, make training data for discrete GP training
	# GP(X_j) ~= Xtrue_{j+1} - Psi_slow(Xtrue_j), where Xtrue are true solutions of the slow variable
	foo_nm = 'discrete_training'
	try:
		X_train_gp = master_output_dict[foo_nm+'_X_train_gp']
		y_train_gp = master_output_dict[foo_nm+'_y_train_gp']
		gp_train_inds_full = master_output_dict[foo_nm+'_gp_train_inds_full']
		gp_train_inds_share = master_output_dict[foo_nm+'_gp_train_inds_share']
	except:
		X_train_gp = foo['X_train']
		slow_preds = np.zeros((X_train_gp.shape[0]-1, X_train_gp.shape[1]))
		for j in range(X_train_gp.shape[0]-1):
			ic = X_train_gp[j,:] # initial condition
			sol = solve_ivp(fun=lambda t, y: ODE.slow(y, t), t_span=(0, delta_t), y0=ic, method=psi0_ode_int_method, rtol=psi0_ode_int_rtol, atol=psi0_ode_int_atol, max_step=psi0_ode_int_max_step, t_eval=np.array([delta_t]))
			slow_preds[j,:] = sol.y.T # prediction Psi_slow(Xtrue_j)
		y_train_gp = X_train_gp[1:,:] - slow_preds # get the residuals
		X_train_gp = X_train_gp[:-1,:] # get the inputs
		gp_train_inds_full = get_inds(N_total=X_train_gp.shape[0], N_subsample=n_subsample_gp)
		gp_train_inds_share = get_inds(N_total=X_train_gp.reshape(-1,1).shape[0], N_subsample=n_subsample_gp)
		master_output_dict[foo_nm+'_X_train_gp'] = X_train_gp
		master_output_dict[foo_nm+'_y_train_gp'] = y_train_gp
		master_output_dict[foo_nm+'_gp_train_inds_full'] = gp_train_inds_full
		master_output_dict[foo_nm+'_gp_train_inds_share'] = gp_train_inds_share
		np.savez(master_output_fname,**master_output_dict)

	ax_kde_discrete.set_xlabel(r'$X_k$')
	ax_kde_discrete.set_ylabel('Probability density')
	ax_gp_discrete.plot(X_train_gp.reshape(-1,1), y_train_gp.reshape(-1,1)/delta_t, 'o', markersize=2, color='gray', alpha=0.8, label='Training Data (all)')
	# ax_gp_discrete.plot(X_train_gp[gp_train_inds_full,:].reshape(-1,1), y_train_gp[gp_train_inds_full,:].reshape(-1,1)/delta_t, '+', markersize=3, color='cyan', alpha=0.8, label='GP-full Data (subset={n_subsample_gp})'.format(n_subsample_gp=n_subsample_gp))
	ax_gp_discrete.plot(X[train_inds], np.mean(ODE.hx)*Y_true[train_inds], 'o', markersize=2, color='red', alpha=0.8, label='True Continuous Training Data (subset={n_subsample_gp})'.format(n_subsample_gp=n_subsample_gp))
	ax_gp_discrete.plot(X[train_inds], np.mean(ODE.hx)*Y_inferred[train_inds], '+', linewidth=1, markersize=3, markeredgewidth=1, color='green', alpha=0.8, label='Approximate Continuous Training Data (subset={n_subsample_gp})'.format(n_subsample_gp=n_subsample_gp))
	ax_gp_discrete.plot(X_train_gp.reshape(-1,1)[gp_train_inds_share], y_train_gp.reshape(-1,1)[gp_train_inds_share]/delta_t, '+', linewidth=1, markersize=3, markeredgewidth=1, color='purple', alpha=0.8, label='GP-share Discrete Data (subset={n_subsample_gp})'.format(n_subsample_gp=n_subsample_gp))

	# fig_discrete.suptitle('GPR fits to errors of discrete slow-only forward-map')
	ax_gp_discrete.set_xlabel(r'$X^{(n)}_k$')
	ax_gp_discrete.set_ylabel(r'$[X^{(n+1)}_k - \Psi_0(X^{(n)})_k] / \Delta t$')
	ax_gp_discrete.legend(loc='best', prop={'size': 4})

	fig_discrete.savefig(fname=os.path.join(output_dir,'gp_discrete_fits.png'), dpi=300)
	fig.savefig(fname=os.path.join(output_dir,'continuous_fits.png'), dpi=300)

	# color = color_list[c]
	# print('alpha=',alpha, color)

	c = -1 # keep track of color list

	# intialize gpr
	GP_ker = ConstantKernel(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-10, 1e+6)) + WhiteKernel(1.0, (1e-10, 1e6))
	my_gpr = GaussianProcessRegressor(
		kernel = GP_ker,
		n_restarts_optimizer = n_restarts_optimizer,
		alpha = alpha
	)

	######### GP-fulltofull #######
	foo_nm = 'GP_discrete_full'
	# fit GP to residuals of discrete operator
	c += 1
	color = color_list[c]
	X_pred_outer = np.outer(X_pred,np.ones(K))
	try:
		gpr_discrete_full_mean = master_output_dict[foo_nm+'_mean']
		my_kernel = master_output_dict[foo_nm+'_kernel']
	except:
		gpr_discrete_full = my_gpr.fit(X=X_train_gp[gp_train_inds_full,:], y=y_train_gp[gp_train_inds_full,:]/delta_t)
		my_kernel = my_gpr.kernel_
		gpr_discrete_full_mean = gpr_discrete_full.predict(X_pred_outer, return_std=False) # evaluate at [0,0,0,0], [0.01,0.01,0.01,0.01], etc.
		master_output_dict[foo_nm+'_mean'] = gpr_discrete_full_mean
		master_output_dict[foo_nm+'_kernel'] = my_kernel
		np.savez(master_output_fname,**master_output_dict)

	# plot training data
	# ax_gp_discrete.plot(X_pred_outer.reshape(-1,1), gpr_discrete_full_mean.reshape(-1,1), color=color, linestyle='', marker='+', markeredgewidth=0.1, markersize=3, label=r'$\Phi_{{\theta}}(X^{{(n)}}_k)$ ({kernel})'.format(kernel=my_kernel))
	# ax_gp_discrete.plot(X_pred_outer.reshape(-1,1), gpr_discrete_full_mean.reshape(-1,1), color=color, linestyle='-', linewidth=1, label=r'$\Phi_{{\theta}}(X^{{(n)}}_k)$ ~ GP({kernel})'.format(kernel=my_kernel))
	ax_gp_discrete.legend(loc='best', prop={'size': 5.5})
	ax_kde_discrete.legend(loc='lower center', prop={'size': 8})
	ax_acf_discrete.legend(loc='best', prop={'size': 8})
	fig_discrete.savefig(fname=os.path.join(output_dir,'gp_discrete_fits.png'), dpi=300)
	t0 = time()
	try:
		y_clean = master_output_dict[foo_nm+'_y_clean']
	except:
		# now generate a test trajectory using the learned GPR
		y_clean = np.zeros((len(t_eval), K))
		y_clean[0,:] = y0
		for j in range(len(t_eval)-1):
			ic_discrete = y_clean[j,:]
			sol = solve_ivp(fun=lambda t, y: ODE.slow(y, t), t_span=(0, delta_t), y0=ic_discrete, method=psi0_ode_int_method, rtol=psi0_ode_int_rtol, atol=psi0_ode_int_atol, max_step=psi0_ode_int_max_step, t_eval=[delta_t])
			y_pred = sol.y.squeeze()
			# compute fulltofull GPR correction
			y_pred += delta_t*gpr_discrete_full.predict(ic_discrete.reshape(1,-1), return_std=False).squeeze()
			y_clean[j+1,:] = y_pred
		master_output_dict[foo_nm+'_y_clean'] = y_clean
		np.savez(master_output_fname,**master_output_dict)

	X_test_gpr_discrete_full = y_clean[ntsynch:,:K].reshape(-1, 1)
	T_test_gpr_discrete_full_acf = acf(y_clean[ntsynch:,0], fft=True, nlags=nlags) #look at first component
	print('Generated invariant measure for GP-discrete-full:', (time()-t0)/60,'minutes')
	plot_traj(X_learned=y_clean[:n_plot,:K], plot_fname=os.path.join(output_dir,'trajectory_discrete_fullGP_alpha{alpha}.png'.format(alpha=alpha)))
	sns.kdeplot(X_test_gpr_discrete_full[test_inds].squeeze(), ax=ax_kde_discrete, color=color, linestyle='-', label=r'$X_{{k+1}} = \Psi_0(X_k) + \Phi_{{\theta}}(X_k)$')
	# sns.kdeplot(X_test_gpr_discrete_full[test_inds].squeeze(), ax=ax_kde_discrete, color=color, linestyle='', marker='o', markeredgewidth=1, markersize=2, label=r'$X_{{k+1}} = \Psi_0(X_k) + \Phi_{{\theta}}(X_k)$ ({kernel})'.format(kernel=my_kernel))
	ax_acf_discrete.plot(t_acf_plot, T_test_gpr_discrete_full_acf, color=color, linestyle='-', label=r'$X_{{k+1}} = \Psi_0(X_k) + \Phi_{{\theta}}(X_k)$')
	ax_gp_discrete.legend(loc='best', prop={'size': 5.5})
	ax_kde_discrete.legend(loc='lower center', prop={'size': 8})
	ax_acf_discrete.legend(loc='best', prop={'size': 8})
	fig_discrete.savefig(fname=os.path.join(output_dir,'gp_discrete_fits.png'), dpi=300)

	# run model against test trajectories
	try:
		test_traj_preds = master_output_dict[foo_nm+'_test_traj_preds']
		do_compute = False
	except:
		test_traj_preds = np.zeros(X_test_traj.shape)
		do_compute = True
	for t in range(n_test_traj):
		if do_compute:
			ic_discrete = X_test_traj[t,0,:]
			for j in range(X_test_traj.shape[1]):
				test_traj_preds[t,j,:] = ic_discrete # prediction Psi_slow(Xtrue_j)
				sol = solve_ivp(fun=lambda t, y: ODE.slow(y, t), t_span=(0, delta_t), y0=ic_discrete, method=psi0_ode_int_method, rtol=psi0_ode_int_rtol, atol=psi0_ode_int_atol, max_step=psi0_ode_int_max_step, t_eval=np.array([delta_t]))
				y_pred = sol.y.squeeze()
				# compute fulltofull GPR correction
				y_pred += delta_t*gpr_discrete_full.predict(ic_discrete.reshape(1,-1), return_std=False).squeeze()
				ic_discrete = y_pred
			master_output_dict[foo_nm+'_test_traj_preds'] = test_traj_preds
			np.savez(master_output_fname,**master_output_dict)
		# compute traj error
		tval_foo = traj_div_time(Xtrue=X_test_traj[t,:,:], Xpred=test_traj_preds[t,:,:], delta_t=delta_t, avg_output=avg_output, thresh=t_valid_thresh)
		t_valid['Discrete Full'][t] = tval_foo
	for ax in [ax_tvalid_discrete, ax_tvalid]:
		sns.boxplot(ax=ax, data=pd.DataFrame(t_valid), color='orchid')
		ax.set_xticklabels(ax.get_xticklabels(), rotation=20, horizontalalignment='right', fontsize='small')
		ax.set_ylabel('Validity Time')
	fig_discrete.savefig(fname=os.path.join(output_dir,'gp_discrete_fits.png'), dpi=300)
	fig.savefig(fname=os.path.join(output_dir,'continuous_fits.png'), dpi=300)


	######### GP-share #######
	foo_nm = 'GP_discrete_share'
	c += 1
	color = color_list[c]
	# fit GP to residuals of discrete operator
	try:
		gpr_discrete_share_mean = master_output_dict[foo_nm+'_mean']
		my_kernel = master_output_dict[foo_nm+'_kernel']
	except:
		gpr_discrete_share = my_gpr.fit(X=X_train_gp.reshape(-1,1)[gp_train_inds_share], y=y_train_gp.reshape(-1,1)[gp_train_inds_share]/delta_t)
		my_kernel = my_gpr.kernel_
		gpr_discrete_share_mean = gpr_discrete_share.predict(X_pred, return_std=False)
		master_output_dict[foo_nm+'_mean'] = gpr_discrete_share_mean
		master_output_dict[foo_nm+'_kernel'] = my_kernel
		np.savez(master_output_fname,**master_output_dict)
	# plot training data
	ax_gp_discrete.plot(X_pred, gpr_discrete_share_mean, color=color, linestyle='-', label=r'$\bar{{\Phi}}_{{\theta}}(X^{{(n)}}_k)$ ~ GP({kernel})'.format(kernel=my_kernel))
	ax_gp_discrete.legend(loc='best', prop={'size': 5.5})
	ax_kde_discrete.legend(loc='lower center', prop={'size': 8})
	ax_acf_discrete.legend(loc='best', prop={'size': 8})
	fig_discrete.savefig(fname=os.path.join(output_dir,'gp_discrete_fits.png'), dpi=300)
	t0 = time()
	# now generate a test trajectory using the learned GPR
	try:
		y_clean = master_output_dict[foo_nm+'_y_clean']
	except:
		y_clean = np.zeros((len(t_eval), K))
		y_clean[0,:] = y0
		for j in range(len(t_eval)-1):
			ic_discrete = y_clean[j,:]
			sol = solve_ivp(fun=lambda t, y: ODE.slow(y, t), t_span=(0, delta_t), y0=ic_discrete, method=psi0_ode_int_method, rtol=psi0_ode_int_rtol, atol=psi0_ode_int_atol, max_step=psi0_ode_int_max_step, t_eval=[delta_t])
			y_pred = sol.y.squeeze()
			# compute shared GPR correction
			for k in range(K):
				y_pred[k] += delta_t*gpr_discrete_share.predict(ic_discrete[k].reshape(1,-1), return_std=False)
			y_clean[j+1,:] = y_pred
		master_output_dict[foo_nm+'_y_clean'] = y_clean
		np.savez(master_output_fname,**master_output_dict)
	X_test_gpr_discrete_share = y_clean[ntsynch:,:K].reshape(-1, 1)
	T_test_gpr_discrete_share_acf = acf(y_clean[ntsynch:,0], fft=True, nlags=nlags) #look at first component
	print('Generated invariant measure for GP-discrete-share:', (time()-t0)/60,'minutes')
	plot_traj(X_learned=y_clean[:n_plot,:K], plot_fname=os.path.join(output_dir,'trajectory_discrete_shareGP_alpha{alpha}.png'.format(alpha=alpha)))
	sns.kdeplot(X_test_gpr_discrete_share[test_inds].squeeze(), ax=ax_kde_discrete, color=color, linestyle='-', label=r'$X_{{k+1}} = \Psi_0(X_k) + \bar{{\Phi}}_{{\theta}}(X_k)$')
	ax_acf_discrete.plot(t_acf_plot, T_test_gpr_discrete_share_acf, color=color, label=r'$X_{{k+1}} = \Psi_0(X_k) + \bar{{\Phi}}_{{\theta}}(X_k)$')
	ax_gp_discrete.legend(loc='best', prop={'size': 5.5})
	ax_kde_discrete.legend(loc='lower center', prop={'size': 8})
	ax_acf_discrete.legend(loc='best', prop={'size': 8})
	fig_discrete.savefig(fname=os.path.join(output_dir,'gp_discrete_fits.png'), dpi=300)

	# run model against test trajectories
	try:
		test_traj_preds = master_output_dict[foo_nm+'_test_traj_preds']
		do_compute = False
	except:
		test_traj_preds = np.zeros(X_test_traj.shape)
		do_compute = True
	for t in range(n_test_traj):
		if do_compute:
			ic_discrete = X_test_traj[t,0,:]
			for j in range(X_test_traj.shape[1]):
				test_traj_preds[t,j,:] = ic_discrete # prediction Psi_slow(Xtrue_j)
				sol = solve_ivp(fun=lambda t, y: ODE.slow(y, t), t_span=(0, delta_t), y0=ic_discrete, method=psi0_ode_int_method, rtol=psi0_ode_int_rtol, atol=psi0_ode_int_atol, max_step=psi0_ode_int_max_step, t_eval=np.array([delta_t]))
				y_pred = sol.y.squeeze()
				# compute shared GPR correction
				for k in range(K):
					y_pred[k] += delta_t*gpr_discrete_share.predict(ic_discrete[k].reshape(1,-1), return_std=False)
				ic_discrete = y_pred
			master_output_dict[foo_nm+'_test_traj_preds'] = test_traj_preds
			np.savez(master_output_fname,**master_output_dict)
		# compute traj error
		tval_foo = traj_div_time(Xtrue=X_test_traj[t,:,:], Xpred=test_traj_preds[t,:,:], delta_t=delta_t, avg_output=avg_output, thresh=t_valid_thresh)
		t_valid['Discrete Share'][t] = tval_foo
	for ax in [ax_tvalid_discrete, ax_tvalid]:
		sns.boxplot(ax=ax, data=pd.DataFrame(t_valid), color='orchid')
		ax.set_xticklabels(ax.get_xticklabels(), rotation=20, horizontalalignment='right', fontsize='small')
		ax.set_ylabel('Validity Time')
	fig_discrete.savefig(fname=os.path.join(output_dir,'gp_discrete_fits.png'), dpi=300)
	fig.savefig(fname=os.path.join(output_dir,'continuous_fits.png'), dpi=300)


	# np.savez(os.path.join(output_dir,'test_output_discrete_{alpha}.npz'.format(alpha=alpha)),
	# 	X_test_gpr_discrete_share=X_test_gpr_discrete_share,
	# 	X_test_gpr_discrete_full=X_test_gpr_discrete_full,
	# 	X_test=X_test,
	# 	X_test_null=X_test_null)

	plt.close(fig_discrete)


	# now run continuous RHS learning
	ax_kde.set_ylabel('Probability density')
	ax_kde.set_xlabel(r'$X_k$')
	ax_kde.legend(loc='lower center', prop={'size': 4})
	ax_gp.set_xlabel(r'$X_k$')
	ax_gp.set_ylabel(r'$h_x \bar{Y}_k$')
	ax_gp.legend(loc='best', prop={'size': 5})
	fig.savefig(fname=output_fname, dpi=300)
	# alpha = alpha_list_cont[c]
	# color = color_list[c]
	# print('alpha=',alpha, color)

	c = -1
	# intialize gpr
	GP_ker = ConstantKernel(1.0, (1e-5, 1e5)) * RBF(1.0, (1e-10, 1e+6)) + WhiteKernel(1.0, (1e-10, 1e6))
	my_gpr = GaussianProcessRegressor(
		kernel = GP_ker,
		n_restarts_optimizer = n_restarts_optimizer,
		alpha = alpha
	)

	# fit GPR-Ybartrue to Xk vs Ybar-true
	foo_nm = 'GP_continuous_Ytrue'
	c += 1
	color = color_list[c]
	try:
		gpr_true_mean = master_output_dict[foo_nm+'_mean']
		my_kernel = master_output_dict[foo_nm+'_kernel']
	except:
		gpr_true = my_gpr.fit(X=X[train_inds], y=np.mean(ODE.hx)*Y_true[train_inds])
		my_kernel = my_gpr.kernel_
		gpr_true_mean = gpr_true.predict(X_pred, return_std=False)
		master_output_dict[foo_nm+'_mean'] = gpr_true_mean
		master_output_dict[foo_nm+'_kernel'] = my_kernel
		np.savez(master_output_fname,**master_output_dict)
		# now run gp-corrected ODE
		ODE.set_predictor(gpr_true.predict)
	t0 = time()
	try:
		y_clean = master_output_dict[foo_nm+'_y_clean']
	except:
		sol = solve_ivp(fun=lambda t, y: ODE.regressed(y, t), t_span=(t_eval[0], t_eval[-1]), y0=y0, method=testcontinuous_ode_int_method, rtol=testcontinuous_ode_int_rtol, atol=testcontinuous_ode_int_atol, max_step=testcontinuous_ode_int_max_step, t_eval=t_eval)
		y_clean = sol.y.T
		master_output_dict[foo_nm+'_y_clean'] = y_clean
		np.savez(master_output_fname,**master_output_dict)

	X_test_gpr_true_share = y_clean[ntsynch:,:K].reshape(-1, 1)
	T_test_gpr_true_share_acf = acf(y_clean[ntsynch:,0], fft=True, nlags=nlags) #look at first component
	print('Generated invariant measure for GP-true-share:', (time()-t0)/60,'minutes')
	plot_traj(X_learned=y_clean[:n_plot,:K], plot_fname=os.path.join(output_dir,'trajectory_YbarTrue_alpha{alpha}.png'.format(alpha=alpha)))
	sns.kdeplot(X_test_gpr_true_share[test_inds].squeeze(), ax=ax_kde, color=color, linestyle='-', label='RHS = Slow + GP (True Y-avg)')
	ax_gp.plot(X_pred, gpr_true_mean, color=color, linestyle='-', label='GP (True Y-avg) ({kernel})'.format(kernel=my_kernel))
	ax_acf.plot(t_acf_plot, T_test_gpr_true_share_acf, color=color, label='RHS = Slow + GP (True Y-avg)')
	ax_acf.legend(loc='best', prop={'size': 8})
	ax_gp.legend(loc='best', prop={'size': 5.5})
	ax_kde.legend(loc='lower center', prop={'size': 8})
	fig.savefig(fname=output_fname, dpi=300)

	# run model against test trajectories
	try:
		test_traj_preds = master_output_dict[foo_nm+'_test_traj_preds']
		do_compute = False
	except:
		test_traj_preds = np.zeros(X_test_traj.shape)
		do_compute = True
	for t in range(n_test_traj):
		if do_compute:
			ic = X_test_traj[t,0,:]
			for j in range(X_test_traj.shape[1]):
				test_traj_preds[t,j,:] = ic # prediction Psi_slow(Xtrue_j)
				sol = solve_ivp(fun=lambda t, y: ODE.regressed(y, t), t_span=(0, delta_t), y0=ic, method=testcontinuous_ode_int_method, rtol=testcontinuous_ode_int_rtol, atol=testcontinuous_ode_int_atol, max_step=testcontinuous_ode_int_max_step, t_eval=np.array([delta_t]))
				ic = sol.y.T.squeeze()
			master_output_dict[foo_nm+'_test_traj_preds'] = test_traj_preds
			np.savez(master_output_fname,**master_output_dict)
		# compute traj error
		tval_foo = traj_div_time(Xtrue=X_test_traj[t,:,:], Xpred=test_traj_preds[t,:,:], delta_t=delta_t, avg_output=avg_output, thresh=t_valid_thresh)
		t_valid['Continuous Share Y-True'][t] = tval_foo
	for ax in [ax_tvalid_discrete, ax_tvalid]:
		sns.boxplot(ax=ax, data=pd.DataFrame(t_valid), color='orchid')
		ax.set_xticklabels(ax.get_xticklabels(), rotation=20, horizontalalignment='right', fontsize='small')
		ax.set_ylabel('Validity Time')
	fig_discrete.savefig(fname=os.path.join(output_dir,'gp_discrete_fits.png'), dpi=300)
	fig.savefig(fname=os.path.join(output_dir,'continuous_fits.png'), dpi=300)


	# fit GPR-Ybarpprox to Xk vs Ybar-infer
	foo_nm = 'GP_continuous_Yinfer'
	c += 1
	color = color_list[c]
	try:
		gpr_approx_mean = master_output_dict[foo_nm+'_mean']
		my_kernel = master_output_dict[foo_nm+'_kernel']
	except:
		gpr_approx = my_gpr.fit(X=X[train_inds], y=np.mean(ODE.hx)*Y_inferred[train_inds])
		my_kernel = my_gpr.kernel_
		gpr_approx_mean = gpr_approx.predict(X_pred, return_std=False)
		master_output_dict[foo_nm+'_mean'] = gpr_approx_mean
		master_output_dict[foo_nm+'_kernel'] = my_kernel
		np.savez(master_output_fname,**master_output_dict)
		# now run gp-corrected ODE
		ODE.set_predictor(gpr_approx.predict)
	t0 = time()

	try:
		y_clean = master_output_dict[foo_nm+'_y_clean']
	except:
		sol = solve_ivp(fun=lambda t, y: ODE.regressed(y, t), t_span=(t_eval[0], t_eval[-1]), y0=y0, method=testcontinuous_ode_int_method, rtol=testcontinuous_ode_int_rtol, atol=testcontinuous_ode_int_atol, max_step=testcontinuous_ode_int_max_step, t_eval=t_eval)
		y_clean = sol.y.T
		master_output_dict[foo_nm+'_y_clean'] = y_clean
		np.savez(master_output_fname,**master_output_dict)
	X_test_gpr_approx_share = y_clean[ntsynch:,:K].reshape(-1, 1)
	T_test_gpr_approx_share_acf = acf(y_clean[ntsynch:,0], fft=True, nlags=nlags) #look at first component
	print('Generated invariant measure for GP-approx-share:', (time()-t0)/60,'minutes')
	plot_traj(X_learned=y_clean[:n_plot,:K], plot_fname=os.path.join(output_dir,'trajectory_YbarInfer_alpha{alpha}.png'.format(alpha=alpha)))
	sns.kdeplot(X_test_gpr_approx_share[test_inds].squeeze(), ax=ax_kde, color=color, linestyle='--', label='RHS = Slow + GP (Approx Y-avg)')
	ax_gp.plot(X_pred, gpr_approx_mean, color=color, linestyle='--', label='GP (Inferred Y-avg) ({kernel})'.format(kernel=my_kernel))
	ax_acf.plot(t_acf_plot, T_test_gpr_approx_share_acf, color=color, label='RHS = Slow + GP (Inferred Y-avg)')
	ax_gp.legend(loc='best', prop={'size': 5.5})
	ax_acf.legend(loc='best', prop={'size': 8})
	ax_kde.legend(loc='lower center', prop={'size': 8})
	fig.savefig(fname=output_fname, dpi=300)

	# run model against test trajectories
	try:
		test_traj_preds = master_output_dict[foo_nm+'_test_traj_preds']
		do_compute = False
	except:
		test_traj_preds = np.zeros(X_test_traj.shape)
		do_compute = True
	for t in range(n_test_traj):
		if do_compute:
			ic = X_test_traj[t,0,:]
			for j in range(X_test_traj.shape[1]):
				test_traj_preds[t,j,:] = ic # prediction Psi_slow(Xtrue_j)
				sol = solve_ivp(fun=lambda t, y: ODE.regressed(y, t), t_span=(0, delta_t), y0=ic, method=testcontinuous_ode_int_method, rtol=testcontinuous_ode_int_rtol, atol=testcontinuous_ode_int_atol, max_step=testcontinuous_ode_int_max_step, t_eval=np.array([delta_t]))
				ic = sol.y.T.squeeze()
			master_output_dict[foo_nm+'_test_traj_preds'] = test_traj_preds
			np.savez(master_output_fname,**master_output_dict)
		# compute traj error
		tval_foo = traj_div_time(Xtrue=X_test_traj[t,:,:], Xpred=test_traj_preds[t,:,:], delta_t=delta_t, avg_output=avg_output, thresh=t_valid_thresh)
		t_valid['Continuous Share Y-Approx'][t] = tval_foo
	for ax in [ax_tvalid_discrete, ax_tvalid]:
		sns.boxplot(ax=ax, data=pd.DataFrame(t_valid), color='orchid')
		ax.set_xticklabels(ax.get_xticklabels(), rotation=20, horizontalalignment='right', fontsize='small')
		ax.set_ylabel('Validity Time')
	fig_discrete.savefig(fname=os.path.join(output_dir,'gp_discrete_fits.png'), dpi=300)
	fig.savefig(fname=os.path.join(output_dir,'continuous_fits.png'), dpi=300)

	# save figure after each loop
	# np.savez(os.path.join(output_dir,'test_output_continuous_{alpha}.npz'.format(alpha=alpha)),
	# 		X_test_gpr_true_share=X_test_gpr_true_share,
	# 		X_test_gpr_approx_share=X_test_gpr_approx_share,
	# 		X_test=X_test,
	# 		X_test_null=X_test_null)

	# dont be a slob...close the fig when you're done!
	plt.close(fig)

	##### okay, now lets plot everything on big sucker!

	# initialize big plot
	figbig, axlist = plt.subplots(3,3,figsize=[24,24])

	legsize = 12

	# initialize color dict
	plot_dict = {'RHS = Full Multiscale': {'color':'black', 'linestyle':'-'},
				'RHS = Slow': {'color':'gray', 'linestyle':':'},
				'Discrete Full': {'color':color_list[0], 'linestyle':'--'},
				'Discrete Share': {'color':'purple', 'linestyle':'--'},
				'Continuous Share Y-True': {'color':color_list[2], 'linestyle':'-'},
				'Continuous Share Y-Approx': {'color':color_list[3], 'linestyle':'-'}
			}

	palette = {key: plot_dict[key]['color'] for key in plot_dict}

	data_dict = {'RHS = Full Multiscale': X_test,
				'RHS = Slow': X_test_null,
				'Discrete Full': X_test_gpr_discrete_full,
				'Discrete Share': X_test_gpr_discrete_share,
				'Continuous Share Y-True': X_test_gpr_true_share,
				'Continuous Share Y-Approx': X_test_gpr_approx_share
			}

	acf_dict = {'RHS = Full Multiscale': T_test_acf,
			'RHS = Slow': T_test_null_acf,
			'Discrete Full': T_test_gpr_discrete_full_acf,
			'Discrete Share': T_test_gpr_discrete_share_acf,
			'Continuous Share Y-True': T_test_gpr_true_share_acf,
			'Continuous Share Y-Approx': T_test_gpr_approx_share_acf
		}

	acf_error_dict = {key: np.linalg.norm(acf_dict[key]-T_test_acf) for key in acf_dict}

	t0 = time()
	print('Starting KL computations...')
	my_kl = lambda Xapprox, Xtrue=X_test, test_inds=test_inds: kl4dummies(Xtrue=Xtrue[test_inds], Xapprox=Xapprox[test_inds], gridsize=512)
	kl_dict = {key: my_kl(data_dict[key]) for key in data_dict}
	print('Finished KL computations in', (time()-t0)/60, 'minutes')

	# KDEs
	ax = axlist[1][0]
	for label in data_dict:
		sns.kdeplot(data_dict[label][test_inds].squeeze(), ax=ax, label=label, linewidth=2, **plot_dict[label])
	ax.legend(loc='best', prop={'size': legsize})
	ax.set_title(r'Invariant Measure KDE')
	ax.set_xlabel(r'$X_k$')
	ax.set_ylabel('Probability')
	figbig.savefig(fname=os.path.join(output_dir,'big_summary.png'), dpi=300)

	# KL-div of KDEs
	ax = axlist[2][0]
	# sns.barplot(ax=ax, data=pd.DataFrame(kl_dict), color='lightseagreen')
	sns.barplot(ax=ax, data=pd.DataFrame(kl_dict), palette=palette)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=20, horizontalalignment='right', fontsize='x-large')
	ax.set_ylabel(r'$D_{\mathrm{KL}}$')
	ax.set_title(r'$D_{\mathrm{KL}}(\mathrm{Corrected} \ || \ \mathrm{True})$')
	ax.set_yscale('log')
	figbig.savefig(fname=os.path.join(output_dir,'big_summary.png'), dpi=300)

	# ACF
	ax = axlist[1][1]
	for label in acf_dict:
		ax.plot(t_acf_plot, acf_dict[label], label=label, linewidth=2, **plot_dict[label])
	ax.set_title(r'Autocorrelation Function ($X_0$)')
	ax.set_xlabel('Time (lag)')
	ax.set_ylabel('ACF')
	ax.set_xscale('linear')
	ax.legend(loc='best', prop={'size': legsize})
	figbig.savefig(fname=os.path.join(output_dir,'big_summary.png'), dpi=300)

	# ACF-log
	ax = axlist[1][2]
	for label in acf_dict:
		ax.plot(t_acf_plot, acf_dict[label], label=label, linewidth=2, **plot_dict[label])
	ax.set_title(r'Autocorrelation Function ($X_0$)')
	ax.set_xlabel('Time (lag)')
	ax.set_ylabel('ACF')
	ax.set_xscale('log')
	ax.legend(loc='best', prop={'size': legsize})
	figbig.savefig(fname=os.path.join(output_dir,'big_summary.png'), dpi=300)

	# ACF-error
	ax = axlist[2][1]
	sns.barplot(ax=ax, data=pd.DataFrame(acf_error_dict, index=[0]), palette=palette)
	# sns.barplot(ax=ax, data=pd.DataFrame(acf_error_dict, index=[0]), color='skyblue')
	ax.set_xticklabels(ax.get_xticklabels(), rotation=20, horizontalalignment='right', fontsize='x-large')
	ax.set_title(r'$||\mathrm{ACF}_{\mathrm{True}} - \mathrm{ACF}_{\mathrm{Corrected}}||$')
	ax.set_ylabel('ACF Error')
	figbig.savefig(fname=os.path.join(output_dir,'big_summary.png'), dpi=300)

	# T-valid box plot
	ax = axlist[2][2]
	sns.boxplot(ax=ax, data=pd.DataFrame(t_valid), palette=palette)
	# sns.boxplot(ax=ax, data=pd.DataFrame(t_valid), color='orchid')
	ax.set_xticklabels(ax.get_xticklabels(), rotation=20, horizontalalignment='right', fontsize='x-large')
	ax.set_ylabel(r'$\tau$')
	ax.set_title('Validity Times')
	figbig.savefig(fname=os.path.join(output_dir,'big_summary.png'), dpi=300)


	# GP-continuous fit
	ax = axlist[0][0]
	ax.plot(X, np.mean(ODE.hx)*Y_true, 'o', markersize=2, color='gray', alpha=0.8, label='True Training Data (all)')
	ax.plot(X[train_inds], np.mean(ODE.hx)*Y_true[train_inds], 'o', markersize=2, color='red', alpha=0.8, label='True Training Data (subset={n_subsample_gp})'.format(n_subsample_gp=n_subsample_gp))
	ax.plot(X[train_inds], np.mean(ODE.hx)*Y_inferred[train_inds], '+', linewidth=1, markersize=3, markeredgewidth=1, color='green', alpha=0.8, label='Approximate Training Data (subset={n_subsample_gp})'.format(n_subsample_gp=n_subsample_gp))
	ax.plot(X_pred, gpr_true_mean, label='Continuous Share Y-True', linewidth=2, **plot_dict['Continuous Share Y-True'])
	ax.plot(X_pred, gpr_approx_mean, label='Continuous Share Y-Approx', linewidth=2, **plot_dict['Continuous Share Y-Approx'])
	ax.set_xlabel(r'$X_k$')
	ax.set_ylabel(r'$h_x \bar{Y}_k$')
	ax.set_title('Continuous Setting')
	ax.legend(loc='best', prop={'size': legsize})
	figbig.savefig(fname=os.path.join(output_dir,'big_summary.png'), dpi=300)


	# GP-discrete fit
	ax = axlist[0][1]
	ax.plot(X_train_gp.reshape(-1,1), y_train_gp.reshape(-1,1)/delta_t, 'o', markersize=2, color='gray', alpha=0.8, label='Training Data (all)')
	# ax.plot(X[train_inds], np.mean(ODE.hx)*Y_true[train_inds], 'o', markersize=2, color='red', alpha=0.8, label='True Continuous Training Data (subset={n_subsample_gp})'.format(n_subsample_gp=n_subsample_gp))
	# ax.plot(X[train_inds], np.mean(ODE.hx)*Y_inferred[train_inds], '+', linewidth=1, markersize=3, markeredgewidth=1, color='green', alpha=0.8, label='Approximate Continuous Training Data (subset={n_subsample_gp})'.format(n_subsample_gp=n_subsample_gp))
	ax.plot(X_train_gp.reshape(-1,1)[gp_train_inds_share], y_train_gp.reshape(-1,1)[gp_train_inds_share]/delta_t, 'o', markersize=2, color=color_list[1], alpha=0.8, label='GP-share Discrete Data (subset={n_subsample_gp})'.format(n_subsample_gp=n_subsample_gp))
	ax.plot(X_pred, gpr_discrete_share_mean, label='Discrete Share', linewidth=2, **plot_dict['Discrete Share'])
	ax.set_xlabel(r'$X^{(n)}_k$')
	ax.set_ylabel(r'$[X^{(n+1)}_k - \Psi_0(X^{(n)})_k] / \Delta t$')
	ax.set_title('Discrete Setting')
	ax.legend(loc='best', prop={'size': legsize})
	figbig.savefig(fname=os.path.join(output_dir,'big_summary.png'), dpi=300)


	# all GP means
	ax = axlist[0][2]
	label = 'Continuous Share Y-True'
	ax.plot(X_pred, gpr_true_mean, label=label, linewidth=2, **plot_dict[label])
	label = 'Continuous Share Y-Approx'
	ax.plot(X_pred, gpr_approx_mean, label=label, linewidth=2, **plot_dict[label])
	label = 'Discrete Share'
	ax.plot(X_pred, gpr_discrete_share_mean, label=label, linewidth=2, **plot_dict[label])
	ax.set_xlabel(r'$X^{(n)}_k$')
	ax.set_title('GP means')
	ax.legend(loc='best', prop={'size': legsize})
	figbig.savefig(fname=os.path.join(output_dir,'big_summary.png'), dpi=300)


	# now, spruce up all the font sizes
	for i in range(len(axlist)):
		for ax in axlist[i]:
			ax.title.set_fontsize(28)
			ax.yaxis.label.set_fontsize(20)
			ax.xaxis.label.set_fontsize(20)
			ax.tick_params(labelsize=16, width=1.5)
			for axis in ['top','bottom','left','right']:
				ax.spines[axis].set_linewidth(2)

	figbig.subplots_adjust(wspace=0.3, hspace=0.3)
	figbig.savefig(fname=os.path.join(output_dir,'big_summary.png'), dpi=300)

	plt.close(figbig)

	return


if __name__ == '__main__':
	continuous_fits()
	run_traintest()


