## This script does the following
#1. Generates a random input sequence x
#2. Simulates data using a driven exponential decay ODE model
#3. Trains a single-layer RNN using clean data output from ODE and the input sequence
#4. RESULT: Gradients quickly go to 0 when training the RNN

# based off of code from https://www.cpuheater.com/deep-learning/introduction-to-recurrent-neural-networks-in-pytorch/
import os
from time import time
from datetime import timedelta
import math
import numpy as np
import numpy.matlib
from scipy.stats import entropy
from scipy.integrate import odeint, solve_ivp
# from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.stats import gaussian_kde
import scipy.optimize

from sklearn.gaussian_process import GaussianProcessRegressor

import torch
from torch.autograd import Variable
import torch.nn.init as init
import torch.cuda

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import pdb


LORENZ_DEFAULT_PARAMS = (10, 28, 8/3)

def str2array(x):
	# x = '[[1,0,0],[0,1,0],[0,0,1]]'
	y = np.array([[float(d) for d in c] for c in [b.split(',') for b in [a.strip('[').strip(']') for a in x.split('],[')]]])
	return y

def get_lorenz_inits(model=None, params=None, n=2):
	init_vec = np.zeros((n, 3))
	for k in range(n):
		(xmin, xmax) = (-10,10)
		(ymin, ymax) = (-20,30)
		(zmin, zmax) = (10,40)

		xrand = xmin+(xmax-xmin)*np.random.random()
		yrand = ymin+(ymax-ymin)*np.random.random()
		zrand = zmin+(zmax-zmin)*np.random.random()
		init_vec[k,:] = [xrand, yrand, zrand]
	return init_vec


def kde_scipy(x, x_grid, **kwargs):
	"""Kernel Density Estimation with Scipy"""
	# Note that scipy weights its bandwidth by the covariance of the
	# input data.  To make the results comparable to the other methods,
	# we divide the bandwidth by the sample standard deviation here.
	kde = gaussian_kde(x, **kwargs)
	return kde.evaluate(x_grid)

# def d(x, x_grid, **kwargs):
#     """Univariate Kernel Density Estimation with Statsmodels.
#     For more options, see
#     https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation."""
#     #statsmodels.nonparametric.kde NOT available on HPC python/2.7.15-tf module.
#     kde = KDEUnivariate(x)
#     kde.fit(**kwargs)
#     return kde.evaluate(x_grid)

def kl4dummies(Xtrue, Xapprox, kde_func=kde_scipy):
	n_states = Xtrue.shape[1]
	kl_vec = np.zeros(n_states)
	for i in range(n_states):
		zmin = min(min(Xtrue[:,i]), min(Xapprox[:,i]))
		zmax = max(max(Xtrue[:,i]), max(Xapprox[:,i]))
		x_grid = np.linspace(zmin, zmax, 10000)
		Pk = kde_func(Xapprox[:,i].astype(np.float), x_grid) # P is approx dist
		Qk = kde_func(Xtrue[:,i].astype(np.float), x_grid) # Q is reference dist
		kl_vec[i] = entropy(Pk, Qk) # compute Dkl(P | Q)
	return kl_vec

### ODE simulation section
## 1. Simulate ODE
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def oscillator_2d(Y,t,a,b,c):
	(x,y) = Y
	dxdt = a*x + b*y -c*x*( x**2 + y**2 )
	dydt = -a*x + b*y -c*y*( x**2 + y**2 )
	dYdt = [dxdt, dydt]
	return dYdt

def harmonic_motion(Y,t,k,m):
	(x,v) = Y
	dxdt = v
	dvdt = -k/m*x
	dYdt = [dxdt, dvdt]
	return dYdt

def single_well(y,t,a,b):
	dydt = -a*y + b
	return dydt

def sine_wave(y,t,a,b):
	dydt = -a*np.cos(b*y)
	return dydt

def double_well(y,t,a,b,c):
	dydt = a*y - b*y**3 + c
	return dydt

# function that returns dy/dt
def exp_decay_model(y,t,yb,c_gamma,x):
	x_t = x[np.where(x[:,0] <= t)[0][-1], 1]
	dydt = -c_gamma*(y-yb) + x_t
	return dydt

def easy_exp_decay_model(y_in,t,yb,c_gamma,x_in):
	dydt = -c_gamma*(y_in-yb) + x_in[0]
	return dydt

def lorenz63(Y,t,a,b,c):
	(x,y,z) = Y
	dxdt = -a*x + a*y
	dydt = b*x - y - x*z
	dzdt = -c*z + x*y

	dYdt = [dxdt, dydt, dzdt]
	return dYdt

def lorenz63_perturbed(Y,t,a=10,b=28,c=8/3,gamma=1,delta=0):
	(x,y,z) = Y
	x = x + delta*(np.sin(gamma*x*y*z))
	y = y + delta*(np.sin(gamma*x*y*z))
	z = z + delta*(np.sin(gamma*x*y*z))
	dxdt = -a*x + a*y
	dydt = b*x - y - x*z
	dzdt = -c*z + x*y

	dYdt = [dxdt, dydt, dzdt]
	return dYdt

def f_normalize_ztrans(norm_dict,y):
	y_norm = (y - norm_dict['Xmean']) / norm_dict['Xsd']
	return y_norm

def f_unNormalize_ztrans(norm_dict,y_norm):
	y = norm_dict['Xsd']*y_norm + norm_dict['Xmean']
	return y

def f_normalize_minmax(norm_dict,y):
	y_norm = (y - norm_dict['Ymin']) / (norm_dict['Ymax'] - norm_dict['Ymin'])
	return y_norm

def f_unNormalize_minmax(norm_dict,y_norm):
	# foo = np.matlib.repmat(norm_dict['Ymax'] - norm_dict['Ymin'], y_norm.shape[0], 1)
	# y = norm_dict['Ymin'] + y_norm * foo
	y = norm_dict['Ymin'] + y_norm * (norm_dict['Ymax'] - norm_dict['Ymin'])
	return y

def run_ode_model(model, tspan, sim_model_params, tau=50, noise_frac=0, output_dir=".", drive_system=True, plot_state_indices=None):
	# time points
	# tau = 50 # window length of persistence

	if drive_system:
		# tmp = np.arange(0,1000,tau)
		tmp = np.arange(0,tspan[-1],tau)
		x = np.zeros([len(tmp),2])
		x[:,0] = tmp
		x[:,1] = 0*10*np.random.rand(len(x))
		my_args = sim_model_params['ode_params'] + (x,)
	else:
		my_args = sim_model_params['ode_params']
		x = None

	y0 = sim_model_params['state_init']
	# y_clean = odeint(model, y0, tspan, args=my_args, mxstep=sim_model_params['mxstep'])

	# pdb.set_trace()
	sol = solve_ivp(fun=lambda t, y: model(y, t, *my_args), t_span=(tspan[0], tspan[-1]), y0=np.array(y0).T, method='RK45', t_eval=tspan)
	y_clean = sol.y.T


	# CHOOSE noise size based on range of the data...if you base on mean or mean(abs),
	# can get disproportionate SDs if one state oscillates between pos/neg, and another state is always POS.
	y_noisy = y_clean + noise_frac*(np.max(y_clean,0) - np.min(y_clean,0))*np.random.randn(len(y_clean),y_clean.shape[1])

	if not plot_state_indices:
		plot_state_indices = np.arange(y_noisy.shape[1])

	## Plot clean ODE simulation
	fig, (ax_list) = plt.subplots(len(plot_state_indices),1)
	if not isinstance(ax_list,np.ndarray):
		ax_list = [ax_list]
	for kk in range(len(ax_list)):
		ax = ax_list[kk]
		ax.plot(tspan, y_clean[:,plot_state_indices[kk]], label='clean data')
		ax.scatter(tspan, y_noisy[:,plot_state_indices[kk]], color='red', s=10, alpha=0.3, label='noisy data')
		ax.set_xlabel('time')
		ax.set_ylabel(sim_model_params['state_names'][plot_state_indices[kk]] +'(t)')
		# ax.tick_params(axis='y')
		# ax.set_title('Testing Fit')
	ax_list[0].legend()
	fig.suptitle('ODE simulation (clean)')
	fig.savefig(fname=output_dir+'/ODEsimulation')
	plt.close(fig)

	# ## 2. Plot ODE
	# fig, ax1 = plt.subplots()

	# color = 'tab:red'
	# ax1.scatter(tspan, y_noisy[:,0], s=10, alpha=0.3, color=color, label='noisy simulation')
	# ax1.plot(tspan, y_clean, color=color, label='clean simulation')
	# ax1.set_xlabel('time')
	# ax1.set_ylabel('y(t)', color=color)
	# ax1.tick_params(axis='y', labelcolor=color)

	# if drive_system:
	# 	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
	# 	color = 'tab:blue'
	# 	ax2.set_ylabel('x(t)', color=color)  # we already handled the x-label with ax1
	# 	ax2.step(x[:,0], x[:,1], ':', where='post', color=color, linestyle='--', label='driver/input data')
	# 	ax2.tick_params(axis='y', labelcolor=color)

	# fig.legend()
	# fig.suptitle('ODE simulation (clean)')
	# fig.savefig(fname=output_dir+'/ODEsimulation_clean')
	# plt.close(fig)

	return y_clean, y_noisy, x


def make_RNN_data(model, tspan, sim_model_params, noise_frac=0, output_dir=".", drive_system=True, plot_state_indices=None):

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	t0 = time()

	# make denser tspan
	tspan_dense = np.linspace(tspan[0],tspan[-1],np.ceil((tspan[-1] - tspan[0])/sim_model_params['smaller_delta_t']+1))

	# intersect tspan_dense with original tspan, then get original indices
	tspan_solve = np.union1d(tspan_dense, tspan)

	ind_orig_tspan = np.searchsorted(tspan_solve, tspan)
	if not np.array_equal(tspan_solve[ind_orig_tspan],tspan):
		raise ValueError('BUG IN THE CODE with subsetting tspan')

	y_clean_dense, y_noisy_dense, x_dense  = run_ode_model(model, tspan_solve, sim_model_params, noise_frac=noise_frac, output_dir=output_dir, drive_system=drive_system, plot_state_indices=plot_state_indices)

	# find subset of dense
	y_clean = y_clean_dense[ind_orig_tspan,:]
	y_noisy = y_noisy_dense[ind_orig_tspan,:]
	if x_dense:
		x = x_dense[ind_orig_tspan,:]
	else:
		x = x_dense

	print("Took {0} seconds to integrate the ODE.".format(time()-t0))
	if drive_system:
		# little section to upsample the random, piecewise constant x(t) function
		z = np.zeros([len(tspan),2])
		z[:,0] = tspan
		c = 0
		prev = 0
		for i in range(len(tspan)):
			if c < len(x) and z[i,0]==x[c,0]:
				prev = x[c,1]
				c += 1
			z[i,1] = prev
		x0 = z[:,None,1] # add an extra dimension to the ode inputs so that it is a tensor for PyTorch
		input_data = x0.T
	else:
		input_data = None

	return input_data, y_clean, y_noisy


def make_RNN_data2(model, tspan_train, tspan_test, sim_model_params, noise_frac=0, output_dir=".", drive_system=True, plot_state_indices=None, n_test_sets=1, f_get_state_inits=None, continue_trajectory=False):

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	t0 = time()
	init_vec = f_get_state_inits(model=model, params=sim_model_params, n=(n_test_sets+1))

	# first get training set
	sim_model_params['state_init'] = init_vec[0,:]
	# y_clean_train, y_noisy_train, x_train  = run_ode_model(model, tspan_train, sim_model_params, noise_frac=noise_frac, output_dir=output_dir, drive_system=drive_system, plot_state_indices=plot_state_indices)

	# make denser tspan
	tspan_dense = np.linspace(tspan_train[0],tspan_train[-1],np.ceil((tspan_train[-1] - tspan_train[0])/sim_model_params['smaller_delta_t']+1))

	# intersect tspan_dense with original tspan, then get original indices
	tspan_solve = np.union1d(tspan_dense, tspan_train)

	ind_orig_tspan = np.searchsorted(tspan_solve, tspan_train)

	if not np.array_equal(tspan_solve[ind_orig_tspan],tspan_train):
		raise ValueError('BUG IN THE CODE with subsetting tspan')

	y_clean_train_dense, y_noisy_train_dense, x_train_dense  = run_ode_model(model, tspan_solve, sim_model_params, noise_frac=noise_frac, output_dir=output_dir, drive_system=drive_system, plot_state_indices=plot_state_indices)

	y_clean_train = y_clean_train_dense[ind_orig_tspan,:]
	y_noisy_train = y_noisy_train_dense[ind_orig_tspan,:]
	if x_train_dense:
		x_train = x_train_dense[ind_orig_tspan,:]
	else:
		x_train = x_train_dense

	# now, get N test sets
	for n in range(n_test_sets):
		if continue_trajectory:
			# increment 1 time step from end of training data
			y0 = y_clean_train[-1,:]
			tspan = get_tspan(sim_model_params)
			# tspan = [0, 0.5*sim_model_params['delta_t'], sim_model_params['delta_t']]
			y_out = odeint(model, y0, tspan, args=sim_model_params['ode_params'], mxstep=sim_model_params['mxstep'])


			pdb.set_trace()
			sol = solve_ivp(fun=lambda t, y: model(y, t, *sim_model_params['ode_params']), t_span=(tspan[0], tspan[-1]), y0=y0.T, method='RK45', t_eval=tspan)
			y_out2 = sol.y.T


			init_continued = y_out[-1,:]
			sim_model_params['state_init'] = init_continued #y_clean_train[-1,:]
		else:
			sim_model_params['state_init'] = init_vec[n+1,:]
		y_clean_test, y_noisy_test, x_test  = run_ode_model(model, tspan_test, sim_model_params, noise_frac=noise_frac, output_dir=output_dir, drive_system=drive_system, plot_state_indices=plot_state_indices)
		if n==0:
			y_clean_test_vec = np.zeros((n_test_sets,y_clean_test.shape[0],y_clean_test.shape[1]))
			y_noisy_test_vec = np.zeros((n_test_sets,y_noisy_test.shape[0],y_noisy_test.shape[1]))
			if drive_system:
				x_test_vec = np.zeros((n_test_sets,x_test.shape[0],x_test.shape[1]))
			else:
				x_test_vec = None
		y_clean_test_vec[n,:,:] = y_clean_test
		y_noisy_test_vec[n,:,:] = y_noisy_test
		if drive_system:
			x_test_vec[n,:,:] = x_test

	# now create multiple training sets
	print("Took {0} seconds to integrate the ODE.".format(time()-t0))
	if drive_system:
		# little section to upsample the random, piecewise constant x(t) function
		z = np.zeros([len(tspan),2])
		z[:,0] = tspan
		c = 0
		prev = 0
		for i in range(len(tspan)):
			if c < len(x) and z[i,0]==x[c,0]:
				prev = x[c,1]
				c += 1
			z[i,1] = prev
		x0 = z[:,None,1] # add an extra dimension to the ode inputs so that it is a tensor for PyTorch
		input_data_train = x0.T
	else:
		input_data_train = None

	return input_data_train, y_clean_train, y_noisy_train, y_clean_test_vec, y_noisy_test_vec, x_test_vec



### RNN fitting section
def forward_vanilla(data_input, hidden_state, w1, w2, b, c, v, *args, **kwargs):
	solver_failed = False
	hidden_state = torch.relu(b + torch.mm(w2,data_input) + torch.mm(w1,hidden_state))
	out = c + torch.mm(v,hidden_state)
	return  (out, hidden_state, solver_failed)

def forward_chaos_pureML(data_input, hidden_state, A, B, C, a, b, *args, **kwargs):
	solver_failed = False
	hidden_state = torch.relu(a + torch.mm(A,hidden_state) + torch.mm(B,data_input))
	# hidden_state = torch.relu(a + torch.mm(A,hidden_state))
	out = b + torch.mm(C,hidden_state)
	return  (out, hidden_state, solver_failed)

def forward_chaos_pureML2(data_input, hidden_state, A, B, C, a, b, *args, **kwargs):
	solver_failed = False
	hidden_state = torch.tanh(a + torch.mm(A,hidden_state))
	out = b + torch.mm(C,hidden_state)
	return  (out, hidden_state, solver_failed)


def get_tspan(model_params):
	tspan = np.linspace(0,model_params['delta_t'],np.ceil(model_params['delta_t']/model_params['smaller_delta_t']+1))
	if tspan[-1] != model_params['delta_t']:
		raise ValueError('BUG IN THE CODE with computing new tspan')
	return tspan


def forward_chaos_hybrid_full(model_input, hidden_state, A, B, C, a, b, normz_info, model, model_params, model_output=None, solver_failed=False):
	# unnormalize
	# ymin = normz_info['Ymin']
	# ymax = normz_info['Ymax']
	# ymean = normz_info['Ymean']
	# ysd = normz_info['Ysd']
	# xmean = normz_info['Xmean']
	# xsd = normz_info['Xsd']

	# y0 = ymean + hidden_state[0].detach().numpy()*ysd
	# y0 = ymin + ( hidden_state[0].detach().numpy()*(ymax - ymin) )
	y0_normalized = torch.squeeze(model_input).detach()
	if model_output is None:
		# tspan = [0, 0.5*model_params['delta_t'], model_params['delta_t']]
		tspan = get_tspan(model_params)
		# driver = xmean + xsd*model_input.detach().numpy()
		# my_args = model_params + (driver,)
		#
		# unnormalize model_input so that it can go through the ODE solver
		# y0 = f_unNormalize_minmax(normz_info, y0_normalized.numpy())
		# y_out = odeint(model, y0, tspan, args=model_params['ode_params'], mxstep=model_params['mxstep'])
		# # pdb.set_trace()

		# y_pred = y_out[-1,:] #last column
		# y_pred_normalized = f_normalize_minmax(normz_info, y_pred)
		y0 = f_unNormalize_minmax(normz_info, y0_normalized.numpy())
		if not solver_failed:
			# y_out, info_dict = odeint(model, y0, tspan, args=model_params['ode_params'], mxstep=model_params['mxstep'], full_output=True)

			sol = solve_ivp(fun=lambda t, y: model(y, t, *model_params['ode_params']), t_span=(tspan[0], tspan[-1]), y0=y0.T, method='RK45', t_eval=tspan)
			y_out = sol.y.T

			if not sol.success:
				# solver failed
				print('ODE solver has failed at y0=',y0)
				solver_failed = True
		if solver_failed:
			y_pred_normalized = np.copy(y0_normalized.numpy()) # persist previous solution
		else:
			# solver is OKAY--use the solution like a good boy!
			y_pred_normalized = f_normalize_minmax(normz_info, y_out[-1,:])
	else:
		# pdb.set_trace()
		y_pred_normalized = model_output

	# renormalize
	# hidden_state[0] = torch.from_numpy( (y_out[-1] - ymin) / (ymax - ymin) )
	# hidden_state[0] = torch.from_numpy( (y_out[-1] - ymean) / ysd )

	stacked_input = torch.FloatTensor(np.hstack( (y_pred_normalized, y0_normalized) )[:,None])
	hidden_state = torch.relu( a + torch.mm(A,hidden_state) + torch.mm(B,stacked_input) )
	stacked_output = torch.cat( ( torch.FloatTensor(y_pred_normalized[:,None]), hidden_state ) )
	out = model_params['learn_residuals_rnn']*y_pred_normalized + b + torch.mm(C,stacked_output)
	return  (out, hidden_state, solver_failed)


def forward_mech(input, hidden_state, w1, w2, b, c, v, normz_info, model, model_params):
	# unnormalize
	ymin = normz_info['Ymin']
	ymax = normz_info['Ymax']
	# ymean = normz_info['Ymean']
	# ysd = normz_info['Ysd']
	xmean = normz_info['Xmean']
	xsd = normz_info['Xsd']

	# y0 = ymean + hidden_state[0].detach().numpy()*ysd
	y0 = ymin + ( hidden_state[0].detach().numpy()*(ymax - ymin) )
	tspan = [0,0.5,1]
	driver = xmean + xsd*input.detach().numpy()
	my_args = model_params + (driver,)
	y_out = odeint(model, y0, tspan, args=my_args, mxstep=model_params['mxstep'])

	pdb.set_trace()
	sol = solve_ivp(fun=lambda t, y: model(y, t, *my_args), t_span=(tspan[0], tspan[-1]), y0=y0.T, method='RK45', t_eval=tspan)
	y_out2 = sol.y.T

	# renormalize
	hidden_state[0] = torch.from_numpy( (y_out[-1] - ymin) / (ymax - ymin) )
	# hidden_state[0] = torch.from_numpy( (y_out[-1] - ymean) / ysd )

	hidden_state = torch.relu(b + torch.mm(w2,input) + torch.mm(w1,hidden_state))
	out = c + torch.mm(v,hidden_state)
	return  (out, hidden_state)


def run_GP(y_clean_train, y_noisy_train,
			y_clean_test, y_noisy_test,
			y_clean_testSynch, y_noisy_testSynch,
			model,f_unNormalize_Y,
			model_pred,
			train_seq_length,
			test_seq_length,
			output_size,
			avg_output_test,
			avg_output_clean_test,
			normz_info, model_params, model_params_TRUE, random_attractor_points,
			plot_state_indices,
			output_dir,
			n_plttrain,
			n_plttest,
			n_test_sets,
			err_thresh, gp_style=1, gp_only=False,
			GP_grid = False,
			alpha=1e-10):


	if gp_only:
		gp_nm = ''
	else:
		gp_nm = 'GPR{0}_'.format(gp_style)

	do_resid = 1
	y=y_noisy_train[1:]-model_pred
	if gp_style==1:
		X = y_noisy_train[:-1]
	elif gp_style==2:
		X = np.concatenate((y_noisy_train[:-1],model_pred),axis=1)
	elif gp_style==3:
		X = model_pred
	elif gp_style==4:
		do_resid = 0
		X = y_noisy_train[:-1]
		y=y_noisy_train[1:]

	nXDim = X.shape[1]
	nYDim = y.shape[1]

	# NEW. learn residuals with GP
	gpr = GaussianProcessRegressor(alpha=alpha).fit(X=X,y=y)
	print(gp_nm,'Training Score =',gpr.score(X=X,y=y))

	# gpr = GaussianProcessRegressor().fit(X=output_train[:-1],y=output_train[1:]-model_pred)
	gpr_train_predictions_orig = do_resid*model_pred + gpr.predict(X).squeeze()
	# gpr.score(model_pred, y_noisy_train[1:])

	gpr_train_predictions_rerun = np.zeros([train_seq_length-1, output_size])
	pred = y_noisy_train[0,:,None].squeeze()
	# tspan = [0, 0.5*model_params['delta_t'], model_params['delta_t']]
	tspan = get_tspan(model_params)
	# pdb.set_trace()
	solver_failed = False
	for j in range(train_seq_length-1):
		# target = output_test[j,None]
		# target_clean = output_clean_train[j,None]

		# generate next-step ODE model prediction
		# unnormalize model_input so that it can go through the ODE solver
		if do_resid:
			y0 = f_unNormalize_minmax(normz_info, pred)
			if not solver_failed:
				# y_out, info_dict = odeint(model, y0, tspan, args=model_params['ode_params'], mxstep=model_params['mxstep'], full_output=True)

				sol = solve_ivp(fun=lambda t, y: model(y, t, *model_params['ode_params']), t_span=(tspan[0], tspan[-1]), y0=y0.T, method='RK45', t_eval=tspan)
				y_out = sol.y.T

				if not sol.success:
					# solver failed
					print('ODE solver has failed at y0=',y0)
					solver_failed = True
			if solver_failed:
				my_model_pred = np.copy(pred) # persist previous solution
			else:
				# solver is OKAY--use the solution like a good boy!
				my_model_pred = f_normalize_minmax(normz_info, y_out[-1,:])
		else:
			# don't need it anyway, so just make it 0
			my_model_pred = 0

		if gp_style==1:
			x = pred
		elif gp_style==2:
			x = np.concatenate((pred, my_model_pred))
		elif gp_style==3:
			x = my_model_pred
		elif gp_style==4:
			x = pred


		pred = do_resid*my_model_pred + gpr.predict(x.reshape(1, -1), return_std=False).squeeze()

		# compute losses
		# total_loss_test += (pred - target.squeeze()).pow(2).sum()/2
		# total_loss_clean_test += (pred - target_clean.squeeze()).pow(2).sum()/2
		# pw_loss_test[j] = total_loss_test / avg_output_test
		# pw_loss_clean_test[j] = total_loss_clean_test / avg_output_clean_test
		gpr_train_predictions_rerun[j,:] = pred
	# pdb.set_trace()
	# plot training fits
	y_noisy_train_raw = f_unNormalize_Y(normz_info,y_noisy_train)
	y_clean_train_raw = f_unNormalize_Y(normz_info,y_clean_train)
	gpr_train_predictions_orig_raw = f_unNormalize_Y(normz_info,gpr_train_predictions_orig)
	gpr_train_predictions_rerun_raw = f_unNormalize_Y(normz_info,gpr_train_predictions_rerun)

	fig, (ax_list) = plt.subplots(len(plot_state_indices),1)
	if not isinstance(ax_list,np.ndarray):
		ax_list = [ax_list]

	for kk in range(len(ax_list)):
		ax1 = ax_list[kk]
		t_plot = np.arange(0,round(len(y_noisy_train[:n_plttrain,plot_state_indices[kk]])*model_params['delta_t'],8),model_params['delta_t'])
		ax1.scatter(t_plot, y_noisy_train_raw[:n_plttrain,plot_state_indices[kk]], color='red', s=10, alpha=0.3, label='noisy data')
		ax1.plot(t_plot, y_clean_train_raw[:n_plttrain,plot_state_indices[kk]], color='red', label='clean data')
		ax1.plot(t_plot[1:], gpr_train_predictions_orig_raw[:n_plttrain-1,plot_state_indices[kk]], color='black', label='GP orig')
		ax1.plot(t_plot[1:], gpr_train_predictions_rerun_raw[:n_plttrain-1,plot_state_indices[kk]], color='blue', label='GP re-run')
		ax1.set_xlabel('time')
		ax1.set_ylabel(model_params['state_names'][plot_state_indices[kk]] + '(t)', color='red')
		ax1.tick_params(axis='y', labelcolor='red')

	ax_list[0].legend()

	fig.suptitle('{0} Training Fit'.format(gp_nm.rstrip('_')))
	fig.savefig(fname=output_dir+'/{0}train_fit_ode'.format(gp_nm))
	plt.close(fig)



	# initialize Test outputs
	loss_vec_test = np.zeros((1,n_test_sets))
	loss_vec_clean_test = np.zeros((1,n_test_sets))
	pred_validity_vec_test = np.zeros((1,n_test_sets))
	pred_validity_vec_clean_test = np.zeros((1,n_test_sets))

	# loop over test sets
	tspan = get_tspan(model_params)
	# tspan = [0, 0.5*model_params['delta_t'], model_params['delta_t']]
	# pdb.set_trace()
	for kkt in range(n_test_sets):
		# now compute test loss
		total_loss_test = 0
		total_loss_clean_test = 0
		pred = y_noisy_testSynch[kkt,-1,None].squeeze()
		# pred = np.squeeze(y_noisy_testSynch[-1,:,None].T)
		pw_loss_test = np.zeros(test_seq_length)
		pw_loss_clean_test = np.zeros(test_seq_length)
		gpr_test_predictions = np.zeros([test_seq_length, output_size])
		solver_failed = False
		for j in range(test_seq_length):
			target = y_noisy_test[kkt,j,:]
			target_clean = y_clean_test[kkt,j,:]

			# generate next-step ODE model prediction
			# tspan = [0, 0.5*model_params['delta_t'], model_params['delta_t']]
			# unnormalize model_input so that it can go through the ODE solver
			if do_resid:
				y0 = f_unNormalize_minmax(normz_info, pred)
				if not solver_failed:
					# y_out, info_dict = odeint(model, y0, tspan, args=model_params['ode_params'], mxstep=model_params['mxstep'], full_output=True)

					sol = solve_ivp(fun=lambda t, y: model(y, t, *model_params['ode_params']), t_span=(tspan[0], tspan[-1]), y0=y0.T, method='RK45', t_eval=tspan)
					y_out = sol.y.T

					if not sol.success:
						# solver failed
						print('ODE solver has failed at y0=',y0)
						solver_failed = True
						# pdb.set_trace()
				if solver_failed:
					my_model_pred = np.copy(pred) # persist previous normalized solution
					# pdb.set_trace()
				else:
					# solver is OKAY--use the solution like a good boy!
					my_model_pred = f_normalize_minmax(normz_info, y_out[-1,:])
			else:
				# don't need it anyway, so just make it 0
				my_model_pred = 0

			if gp_style==1:
				x = pred
			elif gp_style==2:
				x = np.concatenate((pred, my_model_pred))
			elif gp_style==3:
				x = my_model_pred
			elif gp_style==4:
				x = pred

			pred = do_resid*my_model_pred + gpr.predict(x.reshape(1, -1) , return_std=False).squeeze()

			# compute losses
			j_loss = sum((pred - target)**2)
			j_loss_clean = sum((pred - target_clean)**2)
			total_loss_test += j_loss
			total_loss_clean_test += j_loss_clean
			pw_loss_test[j] = j_loss**0.5 / avg_output_test
			pw_loss_clean_test[j] = j_loss_clean**0.5 / avg_output_clean_test
			gpr_test_predictions[j,:] = pred

		total_loss_test = total_loss_test / test_seq_length
		total_loss_clean_test = total_loss_clean_test / test_seq_length

		#store losses
		loss_vec_test[0,kkt] = total_loss_test
		loss_vec_clean_test[0,kkt] = total_loss_clean_test
		pred_validity_vec_test[0,kkt] = np.argmax(pw_loss_test > err_thresh)*model_params['delta_t']
		pred_validity_vec_clean_test[0,kkt] = np.argmax(pw_loss_clean_test > err_thresh)*model_params['delta_t']


		# plot forecast over test set
		y_clean_test_raw = f_unNormalize_Y(normz_info,y_clean_test[kkt,:,:])
		gpr_test_predictions_raw = f_unNormalize_Y(normz_info,gpr_test_predictions)

		fig, (ax_list) = plt.subplots(len(plot_state_indices),1)
		if not isinstance(ax_list,np.ndarray):
			ax_list = [ax_list]

		# NOW, show testing fit
		for kk in range(len(ax_list)):
			ax3 = ax_list[kk]
			t_plot = np.arange(0,len(y_clean_test[kkt,:n_plttest,plot_state_indices[kk]])*model_params['delta_t'],model_params['delta_t'])
			# ax3.scatter(np.arange(len(y_noisy_test)), y_noisy_test, color='red', s=10, alpha=0.3, label='noisy data')
			ax3.plot(t_plot, y_clean_test_raw[:n_plttest,plot_state_indices[kk]], color='red', label='clean data')
			ax3.plot(t_plot, gpr_test_predictions_raw[:n_plttest,plot_state_indices[kk]], color='black', label='NN fit')
			ax3.set_xlabel('time')
			ax3.set_ylabel(model_params['state_names'][plot_state_indices[kk]] +'(t)', color='red')
			ax3.tick_params(axis='y', labelcolor='red')

		ax_list[0].legend()

		fig.suptitle('{0} Testing fit'.format(gp_nm.rstrip('_')))
		fig.savefig(fname=output_dir+'/{0}fit_ode_TEST_{1}'.format(gp_nm,kkt))
		plt.close(fig)

	# pdb.set_trace()
	# write pw losses to file
	# gpr_pred_validity_vec_test = np.argmax(pw_loss_test > err_thresh)*model_params['delta_t']
	# gpr_pred_validity_vec_clean_test = np.argmax(pw_loss_clean_test > err_thresh)*model_params['delta_t']
	# with open(output_dir+'/{0}prediction_validity_time_test.txt'.format(gp_nm), "w") as f:
	# 	f.write(str(gpr_pred_validity_vec_test))
	# with open(output_dir+'/{0}prediction_validity_time_clean_test.txt'.format(gp_nm), "w") as f:
	# 	f.write(str(gpr_pred_validity_vec_clean_test))
	# with open(output_dir+'/{0}loss_test.txt'.format(gp_nm), "w") as f:
	# 	f.write(str(total_loss_test))
	# with open(output_dir+'/{0}clean_loss_test.txt'.format(gp_nm), "w") as f:
	# 	f.write(str(total_loss_clean_test))
	np.savetxt(output_dir+'/{0}loss_vec_test.txt'.format(gp_nm),loss_vec_test)
	np.savetxt(output_dir+'/{0}loss_vec_clean_test.txt'.format(gp_nm),loss_vec_clean_test)
	np.savetxt(output_dir+'/{0}prediction_validity_time_test.txt'.format(gp_nm),pred_validity_vec_test)
	np.savetxt(output_dir+'/{0}prediction_validity_time_clean_test.txt'.format(gp_nm),pred_validity_vec_clean_test)

	if GP_grid:
		## Evaluate GP vs actual residuals on a BOX
		t0grid = time()
		n_residuals = 5000
		GP_error_grid = np.zeros((n_residuals,n_residuals,n_residuals))
		GP_total_error = 0
		# bigX = np.zeros((len(xvals)*len(yvals)*len(zvals),nXDim))
		# bigY = np.zeros((len(xvals)*len(yvals)*len(zvals),nYDim))
		tspan = [0, 0.5*model_params['delta_t'], model_params['delta_t']]

		# FIRST, find the attractor and have something to sample from
		# initialize at FIXED starting point for TRUE model to get the attractor!!!
		# bigTspan = np.arange(0, 2*n_residuals, 0.1)
		# run_again = True
		# while run_again:
		# 	y_out_ATT, info_dict = odeint(model, np.squeeze(get_lorenz_inits(n=1)), bigTspan, args=model_params['ode_params'], mxstep=model_params['mxstep'], full_output=True)
		# 	if info_dict['message'] == 'Integration successful.':
		# 		run_again = False

		# y_out_ATT = y_out_ATT[int(y_out_ATT.shape[0]/3):,]

		# random_attractor_points

		bigX = np.zeros((n_residuals,nXDim))
		bigY = np.zeros((n_residuals,nYDim))

		my_inds = np.random.randint(low=0, high=random_attractor_points.shape[0]-1, size=n_residuals)
		for n in range(len(my_inds)):
			y0 = random_attractor_points[my_inds[n],:]
			y0_normalized = f_normalize_minmax(normz_info, y0)

			y_out, info_dict = odeint(model, y0, tspan, args=model_params['ode_params'], mxstep=model_params['mxstep'], full_output=True)
			bad_model_pred = f_normalize_minmax(normz_info, y_out[-1,:])

			if gp_style==1:
				x_input = y0_normalized
			elif gp_style==2:
				x_input = np.concatenate((y0_normalized, bad_model_pred))
			elif gp_style==3:
				x_input = bad_model_pred
			elif gp_style==4:
				x_input = y0_normalized

			gp_forecast = do_resid*bad_model_pred + gpr.predict(x_input.reshape(1, -1) , return_std=False).squeeze()

			y_out_TRUE, info_dict = odeint(model, y0, tspan, args=model_params_TRUE['ode_params'], mxstep=model_params['mxstep'], full_output=True)

			pdb.set_trace()
			sol = solve_ivp(fun=lambda t, y: model(y, t, *model_params_TRUE['ode_params']), t_span=(tspan[0], tspan[-1]), y0=y0.T, method='RK45', t_eval=tspan)
			y_out_TRUE2 = sol.y.T

			true_model_pred = f_normalize_minmax(normz_info, y_out_TRUE[-1,:])

			bigX[n,:] = x_input
			bigY[n,:] = (true_model_pred - do_resid*bad_model_pred)

			# try:
			# 	my_score = gpr.score(X=x_input, y = (true_model_pred - do_resid*bad_model_pred))
			# except:
			my_score = np.linalg.norm(gp_forecast - true_model_pred)
			# GP_error_grid[ix,iy,iz] = my_score
			GP_total_error += my_score

		print('ATTRACTOR sampling took',str(timedelta(seconds=time()-t0grid)))
		# plot 1-d marginal errors
		print('Total {0} ATTRACTOR error = {1}'.format(gp_nm,GP_total_error))
		print('{0} ATTRACTOR score = {1}'.format(gp_nm,gpr.score(X=bigX,y=bigY)))

		# np.savez(output_dir+'/{0}GP_error_grid'.format(gp_nm), GP_error_grid=GP_error_grid, xvals=xvals, yvals=yvals, zvals=zvals, GP_total_error=GP_total_error)



		## Evaluate GP vs actual residuals on a BOX
		xvals = np.linspace(-10,10,10)
		yvals = np.linspace(-20,30,10)
		zvals = np.linspace(10,40,10)

		t0grid = time()
		ix = 0
		iy = 0
		iz = 0
		GP_error_grid = np.zeros((len(xvals),len(yvals),len(zvals)))
		GP_total_error = 0
		bigX = np.zeros((len(xvals)*len(yvals)*len(zvals),nXDim))
		bigY = np.zeros((len(xvals)*len(yvals)*len(zvals),nYDim))
		tspan = [0, 0.5*model_params['delta_t'], model_params['delta_t']]
		n = -1
		for ix in range(len(xvals)):
			x = xvals[ix]
			for iy in range(len(yvals)):
				y = yvals[iy]
				for iz in range(len(zvals)):
					n += 1
					z = zvals[iz]

					# FIRST, find the attractor (idk, run for >10 lyapunov times...)
					y_out_INIT, info_dict = odeint(model, np.array([x,y,z]), [0, 5, 10], args=model_params['ode_params'], mxstep=model_params['mxstep'], full_output=True)

					pdb.set_trace()
					sol = solve_ivp(fun=lambda t, y: model(y, t, *model_params['ode_params']), t_span=(0, 10), y0=np.array([x,y,z]).T, method='RK45', t_eval=[0, 5, 10])
					y_out_INIT2 = sol.y.T


					y0 = y_out_INIT[-1,:]
					y0_normalized = f_normalize_minmax(normz_info, y0)
					# now, we assume that y0 is on the attractor

					y_out, info_dict = odeint(model, y0, tspan, args=model_params['ode_params'], mxstep=model_params['mxstep'], full_output=True)

					pdb.set_trace()
					sol = solve_ivp(fun=lambda t, y: model(y, t, *model_params['ode_params']), t_span=(tspan[0], tspan[-1]), y0=y0.T, method='RK45', t_eval=tspan)
					y_out2 = sol.y.T



					bad_model_pred = f_normalize_minmax(normz_info, y_out[-1,:])
					if gp_style==1:
						x_input = y0_normalized
					elif gp_style==2:
						x_input = np.concatenate((y0_normalized, bad_model_pred))
					elif gp_style==3:
						x_input = bad_model_pred
					elif gp_style==4:
						x_input = y0_normalized

					gp_forecast = do_resid*bad_model_pred + gpr.predict(x_input.reshape(1, -1) , return_std=False).squeeze()

					y_out_TRUE, info_dict = odeint(model, y0, tspan, args=model_params_TRUE['ode_params'], mxstep=model_params['mxstep'], full_output=True)

					pdb.set_trace()
					sol = solve_ivp(fun=lambda t, y: model(y, t, *model_params_TRUE['ode_params']), t_span=(tspan[0], tspan[-1]), y0=y0.T, method='RK45', t_eval=tspan)
					y_out_TRUE2 = sol.y.T

					true_model_pred = f_normalize_minmax(normz_info, y_out_TRUE[-1,:])

					bigX[n,:] = x_input
					bigY[n,:] = (true_model_pred - do_resid*bad_model_pred)

					# try:
					# 	my_score = gpr.score(X=x_input, y = (true_model_pred - do_resid*bad_model_pred))
					# except:
					my_score = np.linalg.norm(gp_forecast - true_model_pred)
					GP_error_grid[ix,iy,iz] = my_score
					GP_total_error += my_score

		print('Grid took',str(timedelta(seconds=time()-t0grid)))
		# plot 1-d marginal errors
		print('Total {0} grid error = {1}'.format(gp_nm,GP_total_error))
		print('{0} grid score = {1}'.format(gp_nm,gpr.score(X=bigX,y=bigY)))

		np.savez(output_dir+'/{0}GP_error_grid'.format(gp_nm), GP_error_grid=GP_error_grid, xvals=xvals, yvals=yvals, zvals=zvals, GP_total_error=GP_total_error)


def compare_GPs(output_dir,style_list):

	# 1d-marginals
	fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True)
	for gp_style in style_list:
		gp_nm = 'GPR{0}_'.format(gp_style)
		npzfile = np.load(output_dir+'/{0}GP_error_grid.npz'.format(gp_nm))

		ax1.plot(npzfile['xvals'], np.mean(npzfile['GP_error_grid'], axis=(1,2)), label=gp_nm)
		ax2.plot(npzfile['yvals'], np.mean(npzfile['GP_error_grid'], axis=(0,2)), label=gp_nm)
		ax3.plot(npzfile['zvals'], np.mean(npzfile['GP_error_grid'], axis=(0,1)), label=gp_nm)

	ax1.set_xlabel('x')
	ax2.set_xlabel('y')
	ax3.set_xlabel('z')

	ax1.set_ylabel('MSE')
	ax2.set_ylabel('MSE')
	ax3.set_ylabel('MSE')

	ax3.legend()

	fig.suptitle('1-d Marginalized Squared Error of GP-approximated residuals')
	fig.savefig(fname=output_dir+'/1d_Maringal_GP_errors')

	ax1.set_yscale('log')
	ax2.set_yscale('log')
	ax3.set_yscale('log')

	fig.savefig(fname=output_dir+'/1d_Maringal_GP_errors_log')

	plt.close(fig)


	# 2d-marginals
	for gp_style in style_list:
		fig, (ax_XY, ax_XZ, ax_YZ) = plt.subplots(1,3)
		gp_nm = 'GPR{0}_'.format(gp_style)
		npzfile = np.load(output_dir+'/{0}GP_error_grid.npz'.format(gp_nm))

		ax_XY.imshow(np.mean(npzfile['GP_error_grid'], axis=2), label=gp_nm)
		ax_XZ.imshow(np.mean(npzfile['GP_error_grid'], axis=1), label=gp_nm)
		ax_YZ.imshow(np.mean(npzfile['GP_error_grid'], axis=0), label=gp_nm)

		ax_XY.set_xlabel('X')
		ax_XY.set_ylabel('Y')

		ax_XZ.set_xlabel('X')
		ax_XZ.set_ylabel('Z')

		ax_YZ.set_xlabel('Y')
		ax_YZ.set_ylabel('Z')


		fig.suptitle('2-d Marginalized Squared Error of GP-approximated residuals')
		fig.savefig(fname=output_dir+'/2d_Maringal_GP_errors_{0}'.format(gp_nm))
		plt.close(fig)



def train_chaosRNN(forward,
			y_clean_train, y_noisy_train,
			y_clean_test, y_noisy_test,
			y_clean_testSynch, y_noisy_testSynch,
			model_params, hidden_size=6, n_epochs=100, lr=0.05,
			output_dir='.', normz_info=None, model=None,
			trivial_init=False, perturb_trivial_init=True, sd_perturb = 0.001,
			stack_hidden=True, stack_output=True,
			x_train=None, x_test=None,
			f_normalize_Y=f_normalize_minmax,
			f_unNormalize_Y=f_unNormalize_minmax,
			f_normalize_X = f_normalize_ztrans,
			f_unNormalize_X = f_unNormalize_ztrans,
			max_plot=None, n_param_saves=None,
			err_thresh=0.4, plot_state_indices=None,
			precompute_model=True, kde_func=kde_scipy,
			compute_kl=False, gp_only=False, gp_style=None,
			save_iterEpochs=False,
			model_params_TRUE=None,
			GP_grid = False,
			alpha_list = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]):


	t0 = time()

	if torch.cuda.is_available():
		print('Using CUDA FloatTensor')
		dtype = torch.cuda.FloatTensor
	else:
		print('Using regular torch.FloatTensor')
		dtype = torch.FloatTensor

	if max_plot is None:
		max_plot = int(np.floor(30./model_params['delta_t']))

	n_plttrain = min(max_plot,y_clean_train.shape[0])
	n_plttest = min(max_plot,y_clean_test.shape[1])

	if not plot_state_indices:
		plot_state_indices = np.arange(y_clean_test.shape[2])

	if not model_params_TRUE:
		model_params_TRUE = model_params.copy()
		model_params_TRUE['ode_params'] = LORENZ_DEFAULT_PARAMS

	# allow for backwards compatibility (ie if this setting is missing)
	if 'learn_residuals_rnn' not in model_params:
		model_params['learn_residuals_rnn'] = False

	# keep_param_history = np.log10( n_epochs * y_clean_train.shape[0] * (hidden_size**2) ) < mem_thresh_order
	# if not keep_param_history:
	# 	print('NOT saving parameter history...would take too much memory!')

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	print('Starting RNN training for: ', output_dir)

	output_train = torch.FloatTensor(y_noisy_train).type(dtype)
	output_clean_train = torch.FloatTensor(y_clean_train).type(dtype)
	output_test = torch.FloatTensor(y_noisy_test).type(dtype)
	output_clean_test = torch.FloatTensor(y_clean_test).type(dtype)
	test_synch_clean = torch.FloatTensor(y_clean_testSynch).type(dtype)
	test_synch_noisy = torch.FloatTensor(y_noisy_testSynch).type(dtype)

	try:
		avg_output_test = model_params['time_avg_norm']
	except:
		avg_output_test = torch.mean(output_test**2).detach().numpy()**0.5
	# avg_output_test = torch.mean(output_test**2,dim=(0,1)).detach().numpy()**0.5
	try:
		avg_output_clean_test = model_params['time_avg_norm']
	except:
		avg_output_clean_test = torch.mean(output_clean_test**2).detach().numpy()**0.5
	# avg_output_clean_test = torch.mean(output_clean_test**2,dim=(0,1)).detach().numpy()**0.5

	output_size = output_train.shape[1]
	train_seq_length = output_train.size(0)
	test_seq_length = output_test.size(1)
	n_test_sets = output_test.size(0)
	synch_length = test_synch_noisy.size(1)

	# compute one-step-ahead model-based prediction for each point in the training set
	if precompute_model:
		model_pred = np.zeros((train_seq_length-1,output_size))
		for j in range(train_seq_length-1):
			tspan = get_tspan(model_params)
			# tspan = [0, 0.5*model_params['delta_t'], model_params['delta_t']]
			# unnormalize model_input so that it can go through the ODE solver
			y0 = f_unNormalize_minmax(normz_info, output_train[j,:].numpy())
			sol = solve_ivp(fun=lambda t, y: model(y, t, *model_params['ode_params']), t_span=(tspan[0], tspan[-1]), y0=y0.T, method='RK45', t_eval=tspan)

			# y_out = odeint(model, y0, tspan, args=model_params['ode_params'], mxstep=model_params['mxstep'])
			# model_pred[j,:] = f_normalize_minmax(normz_info, y_out[-1,:])
			model_pred[j,:] = f_normalize_minmax(normz_info, sol.y[:,-1])
	else:
		model_pred = [None for j in range(train_seq_length-1)]


	if gp_style is None:
		style_list = [1,2,3,4]
	else:
		style_list = [gp_style]

	# pdb.set_trace()
	# identify Attractor points for GP grid eval
	n_points = 1000
	bigTspan = np.arange(0, 10*n_points, 0.1)
	run_again = True
	while run_again:
		y_out_ATT, info_dict = odeint(model, np.squeeze(get_lorenz_inits(n=1)), bigTspan, args=model_params['ode_params'], mxstep=model_params['mxstep'], full_output=True)
		if info_dict['message'] == 'Integration successful.':
			run_again = False

	# do a one 1/10 burn-in, then randomly downsample.
	my_inds = np.random.randint(low=n_points, high=(10*n_points)-1, size=n_points)
	random_attractor_points = y_out_ATT[my_inds,]

	for gp_style in style_list:
		for alpha in alpha_list:
			print('Running GPR',gp_style,'for alpha=',alpha)
			run_GP(y_clean_train, y_noisy_train,
					y_clean_test, y_noisy_test,
					y_clean_testSynch, y_noisy_testSynch,
					model,f_unNormalize_Y,
					model_pred,
					train_seq_length,
					test_seq_length,
					output_size,
					avg_output_test,
					avg_output_clean_test,
					normz_info, model_params, model_params_TRUE, random_attractor_points,
					plot_state_indices,
					output_dir,
					n_plttrain,
					n_plttest,
					n_test_sets,
					err_thresh,
					gp_style,
					gp_only,
					GP_grid = GP_grid,
					alpha = alpha)

	if gp_only:
		return

	# plot GP comparisons
	if style_list and GP_grid:
		compare_GPs(output_dir,style_list)


	# first, SHOW that a simple mechRNN can fit the data perfectly (if we are running a mechRNN)
	if stack_hidden or stack_output:
		# now, TRAIN to fit the output from the previous model
		# w2 = torch.zeros(hidden_size, input_size).type(dtype)
		A = torch.zeros(hidden_size, hidden_size).type(dtype)
		B = torch.zeros(hidden_size, (1+stack_hidden)*output_size).type(dtype)
		a = torch.zeros(hidden_size, 1).type(dtype)
		C = torch.zeros(output_size, hidden_size + (stack_output*output_size) ).type(dtype)
		b = torch.zeros(output_size, 1).type(dtype)

		for jj in range(output_size):
			C[jj,jj] = 1


		perfect_loss_vec_test = np.zeros((1,n_test_sets))
		perfect_loss_vec_clean_test = np.zeros((1,n_test_sets))
		perfect_pred_validity_vec_test = np.zeros((1,n_test_sets))
		perfect_pred_validity_vec_clean_test = np.zeros((1,n_test_sets))


		perfect_test_loss = []
		perfect_t_valid = []
		for kkt in range(n_test_sets):
			# first, synchronize the hidden state
			hidden_state = torch.zeros((hidden_size, 1)).type(dtype)
			solver_failed = False
			for i in range(synch_length-1):
				(pred, hidden_state, solver_failed) = forward(test_synch_noisy[kkt,i,:,None], hidden_state, A,B,C,a,b , normz_info, model, model_params, solver_failed=solver_failed)

			# hidden_state = torch.zeros((hidden_size, 1)).type(dtype)
			predictions = np.zeros([test_seq_length, output_size])
			# yb_normalized = (yb - YMIN)/(YMAX - YMIN)
			# initializing y0 of hidden state to the true initial condition from the clean signal
			# hidden_state[0] = float(y_clean_test[0])
			# pred = output_train[-1,:,None]
			pred = test_synch_noisy[kkt,-1,:,None]
			perf_total_loss_test = 0
			perf_total_loss_clean_test = 0
			# running_epoch_loss_test = np.zeros(test_seq_length)
			perf_pw_loss_test = np.zeros(test_seq_length)
			perf_pw_loss_clean_test = np.zeros(test_seq_length)
			solver_failed = False
			for i in range(test_seq_length):
				(pred, hidden_state, solver_failed) = forward(pred, hidden_state, A,B,C,a,b , normz_info, model, model_params, solver_failed=solver_failed)
				# hidden_state = hidden_state
				predictions[i,:] = pred.data.numpy().ravel()
				i_loss = (pred.detach().squeeze() - output_test[kkt,i,None].squeeze()).pow(2).sum()
				i_loss_clean = (pred.detach().squeeze() - output_clean_test[kkt,i,None].squeeze()).pow(2).sum()
				perf_total_loss_test += i_loss
				perf_total_loss_clean_test += i_loss_clean
				# running_epoch_loss_test[j] = total_loss_test/(j+1)
				perf_pw_loss_test[i] = i_loss.pow(0.5).numpy() / avg_output_test
				perf_pw_loss_clean_test[i] = i_loss_clean.pow(0.5).numpy() / avg_output_clean_test

			# get error metrics
			perfect_loss_vec_test[0,kkt] = perf_total_loss_test.numpy() / test_seq_length
			perfect_loss_vec_clean_test[0,kkt] = perf_total_loss_clean_test.numpy() / test_seq_length
			perfect_pred_validity_vec_test[0,kkt] = np.argmax(perf_pw_loss_test > err_thresh)*model_params['delta_t']
			perfect_pred_validity_vec_clean_test[0,kkt] = np.argmax(perf_pw_loss_clean_test > err_thresh)*model_params['delta_t']

			# plot predictions vs truth
			fig, (ax_list) = plt.subplots(len(plot_state_indices),1)
			if not isinstance(ax_list,np.ndarray):
				ax_list = [ax_list]

			for kk in range(len(ax_list)):
				ax1 = ax_list[kk]
				ax1.scatter(np.arange(len(y_noisy_test[kkt,:n_plttest,plot_state_indices[kk]])), y_noisy_test[kkt,:n_plttest,plot_state_indices[kk]], color='red', s=10, alpha=0.3, label='noisy data')
				ax1.plot(y_clean_test[kkt,:n_plttest,kk], color='red', label='true model')
				ax1.plot(predictions[:n_plttest,kk], ':' ,color='red', label='perturbed model')
				ax1.set_xlabel('time')
				ax1.set_ylabel(model_params['state_names'][kk] + '(t)', color='red')
				ax1.tick_params(axis='y', labelcolor='red')

			ax_list[0].legend()
			fig.suptitle('RNN w/ just mechanism fit to ODE simulation TEST SET')
			fig.savefig(fname=output_dir+'/PERFECT_MechRnn_fit_TestODE_'+str(kkt))
			plt.close(fig)

		# output perfect fits
		np.savetxt(output_dir+'/perfectModel_loss_vec_test.txt',perfect_loss_vec_test)
		np.savetxt(output_dir+'/perfectModel_loss_vec_clean_test.txt',perfect_loss_vec_clean_test)
		np.savetxt(output_dir+'/perfectModel_prediction_validity_time_test.txt',perfect_pred_validity_vec_test)
		np.savetxt(output_dir+'/perfectModel_prediction_validity_time_clean_test.txt',perfect_pred_validity_vec_clean_test)


	# Initilize parameters for training
	A = torch.zeros(hidden_size, hidden_size).type(dtype)
	B = torch.zeros(hidden_size, (1+stack_hidden)*output_size).type(dtype)
	a = torch.zeros(hidden_size, 1).type(dtype)
	C = torch.zeros(output_size, hidden_size + (stack_output*output_size) ).type(dtype)
	b = torch.zeros(output_size, 1).type(dtype)

	# pdb.set_trace()
	# trivial_init trains the mechRNN starting from parameters (specified above)
	# that trivialize the RNN to the forward ODE model
	# now, TRAIN to fit the output from the previous model
	if trivial_init:
		init.zeros_(A)
		init.zeros_(B)
		init.zeros_(C)
		init.zeros_(a)
		init.zeros_(b)
		for jj in range(output_size):
			C[jj,jj] = 1
		if perturb_trivial_init:
			init.normal_(A, 0.0, sd_perturb)
			init.normal_(B, 0.0, sd_perturb)
			init.normal_(a, 0.0, sd_perturb)
			init.normal_(b, 0.0, sd_perturb)
			C = C + torch.FloatTensor(sd_perturb*np.random.randn(C.shape[0], C.shape[1]))
	else:
		init.normal_(A, 0.0, 0.1)
		init.normal_(B, 0.0, 0.1)
		init.normal_(C, 0.0, 0.1)
		init.normal_(a, 0.0, 0.1)
		init.normal_(b, 0.0, 0.1)

	# additional perturbation for trivial init case


	A = Variable(A, requires_grad=True)
	B =  Variable(B, requires_grad=True)
	C =  Variable(C, requires_grad=True)
	a =  Variable(a, requires_grad=True)
	b =  Variable(b, requires_grad=True)

	if not n_param_saves:
		n_param_saves = int(n_epochs*(train_seq_length-1))
		save_interval = 1
	else:
		n_param_saves = min(n_param_saves, int(n_epochs*(train_seq_length-1)))
		save_interval = int(math.ceil((n_epochs*(train_seq_length-1)/n_param_saves) / 100.0)) * 100

	# if keep_param_history:
	# A_history = np.zeros((n_param_saves, A.shape[0], A.shape[1]))
	# B_history = np.zeros((n_param_saves, B.shape[0], B.shape[1]))
	# C_history = np.zeros((n_param_saves, C.shape[0], C.shape[1]))
	# a_history = np.zeros((n_param_saves, a.shape[0], a.shape[1]))
	# b_history = np.zeros((n_param_saves, b.shape[0], b.shape[1]))
	A_history = np.zeros((n_param_saves,1))
	B_history = np.zeros((n_param_saves,1))
	C_history = np.zeros((n_param_saves,1))
	a_history = np.zeros((n_param_saves,1))
	b_history = np.zeros((n_param_saves,1))
	# A_history_running = np.zeros((n_param_saves,1))
	# B_history_running = np.zeros((n_param_saves,1))
	# C_history_running = np.zeros((n_param_saves,1))
	# a_history_running = np.zeros((n_param_saves,1))
	# b_history_running = np.zeros((n_param_saves,1))

	loss_vec_train = np.zeros(n_epochs)
	loss_vec_clean_train = np.zeros(n_epochs)
	loss_vec_test = np.zeros((n_epochs,n_test_sets))
	loss_vec_clean_test = np.zeros((n_epochs,n_test_sets))
	pred_validity_vec_test = np.zeros((n_epochs,n_test_sets))
	pred_validity_vec_clean_test = np.zeros((n_epochs,n_test_sets))
	kl_vec_inv_test = np.zeros((n_epochs,n_test_sets, output_size))
	kl_vec_inv_clean_test = np.zeros((n_epochs,n_test_sets, output_size))

	cc = -1 # iteration counter for saving weight updates
	cc_inc = -1
	for i_epoch in range(n_epochs):
		total_loss_train = 0
		total_loss_clean_train = 0
		#init.normal_(hidden_state, 0.0, 1)
		#hidden_state = Variable(hidden_state, requires_grad=True)
		pred = output_train[0,:,None]
		hidden_state = Variable(torch.zeros((hidden_size, 1)).type(dtype), requires_grad=True)
		init.normal_(hidden_state,0.0,0.1)
		running_epoch_loss_train = np.zeros(train_seq_length)
		running_epoch_loss_clean_train = np.zeros(train_seq_length)
		solver_failed = False
		for j in range(train_seq_length-1):
			cc += 1
			target = output_train[j+1,None]
			target_clean = output_clean_train[j+1,None]
			# pdb.set_trace()
			(pred, hidden_state, solver_failed) = forward(output_train[j,:,None], hidden_state, A,B,C,a,b, normz_info, model, model_params, model_output=model_pred[j], solver_failed=solver_failed)
			# (pred, hidden_state) = forward(pred.detach(), hidden_state, A,B,C,a,b, normz_info, model, model_params)
			loss = (pred.squeeze() - target.squeeze()).pow(2).sum()
			total_loss_train += loss
			total_loss_clean_train += (pred.squeeze() - target_clean.squeeze()).pow(2).sum()
			running_epoch_loss_train[j] = total_loss_train/(j+1)
			running_epoch_loss_clean_train[j] = total_loss_clean_train/(j+1)
			loss.backward()

			A.data -= lr * A.grad.data
			B.data -= lr * B.grad.data
			C.data -= lr * C.grad.data
			a.data -= lr * a.grad.data
			b.data -= lr * b.grad.data

			# print('|grad_A| = {}'.format(np.linalg.norm(A.grad.data)))
			# print('|grad_B| = {}'.format(np.linalg.norm(B.grad.data)))
			# print('|grad_C| = {}'.format(np.linalg.norm(C.grad.data)))
			# print('|grad_a| = {}'.format(np.linalg.norm(a.grad.data)))
			# print('|grad_b| = {}'.format(np.linalg.norm(b.grad.data)))
			# print("A:",A)
			# print("C:",C)
			# print("a:",a)
			# print("b:",b)

			A.grad.data.zero_()
			B.grad.data.zero_()
			C.grad.data.zero_()
			a.grad.data.zero_()
			b.grad.data.zero_()

			hidden_state = hidden_state.detach()
			# print updates every 2 iterations or in 5% incrememnts
			if (n_epochs==1) and (cc % int( max(2, np.ceil(train_seq_length/20)) ) == 0):
				print("Iteration: {}\nRunning Training Loss = {}\n".format(
							cc,
							running_epoch_loss_train[j]))
			# save updated parameters
			if cc % save_interval == 0:
				cc_inc += 1
				A_history[cc_inc,:] = np.linalg.norm(A.detach().numpy())
				B_history[cc_inc,:] = np.linalg.norm(B.detach().numpy())
				C_history[cc_inc,:] = np.linalg.norm(C.detach().numpy())
				a_history[cc_inc,:] = np.linalg.norm(a.detach().numpy())
				b_history[cc_inc,:] = np.linalg.norm(b.detach().numpy())

				# cumulative means
				# A_history_running[cc_inc,:] = np.mean(A_history[:cc_inc,:])
				# B_history_running[cc_inc,:] = np.mean(B_history[:cc_inc,:])
				# C_history_running[cc_inc,:] = np.mean(C_history[:cc_inc,:])
				# a_history_running[cc_inc,:] = np.mean(a_history[:cc_inc,:])
				# b_history_running[cc_inc,:] = np.mean(b_history[:cc_inc,:])
		#normalize losses
		total_loss_train = total_loss_train / train_seq_length
		total_loss_clean_train = total_loss_clean_train / train_seq_length
		#store losses
		loss_vec_train[i_epoch] = total_loss_train
		loss_vec_clean_train[i_epoch] = total_loss_clean_train

		# now evaluate test loss
		for kkt in range(n_test_sets):
			if synch_length > 1:
				# first, synchronize the hidden state
				hidden_state = torch.zeros((hidden_size, 1)).type(dtype)
				solver_failed = False
				for i in range(synch_length-1):
					(pred, hidden_state, solver_failed) = forward(test_synch_noisy[kkt,i,:,None], hidden_state, A,B,C,a,b , normz_info, model, model_params, solver_failed=solver_failed)
				pred = test_synch_noisy[kkt,-1,:,None]
			else:
				pred = output_train[-1,:,None]

			total_loss_test = 0
			total_loss_clean_test = 0
			running_epoch_loss_test = np.zeros(test_seq_length)
			running_epoch_loss_clean_test = np.zeros(test_seq_length)
			# hidden_state = Variable(torch.zeros((hidden_size, 1)).type(dtype), requires_grad=False)
			pw_loss_test = np.zeros(test_seq_length)
			pw_loss_clean_test = np.zeros(test_seq_length)
			long_predictions = np.zeros([test_seq_length, output_size])
			solver_failed = False
			for j in range(test_seq_length):
				target = output_test[kkt,j,None]
				target_clean = output_clean_test[kkt,j,None]
				(pred, hidden_state, solver_failed) = forward(pred, hidden_state, A,B,C,a,b, normz_info, model, model_params, solver_failed=solver_failed)
				j_loss = (pred.detach().squeeze() - target.squeeze()).pow(2).sum()
				j_loss_clean = (pred.detach().squeeze() - target_clean.squeeze()).pow(2).sum()
				total_loss_test += j_loss
				total_loss_clean_test += j_loss_clean
				running_epoch_loss_clean_test[j] = total_loss_clean_test/(j+1)
				running_epoch_loss_test[j] = total_loss_test/(j+1)
				pw_loss_test[j] = j_loss.pow(0.5).numpy() / avg_output_test
				pw_loss_clean_test[j] = j_loss_clean.pow(0.5).numpy() / avg_output_clean_test
				pred = pred.detach()
				long_predictions[j,:] = pred.data.numpy().ravel()
				hidden_state = hidden_state.detach()

			#normalize losses
			total_loss_test = total_loss_test / test_seq_length
			total_loss_clean_test = total_loss_clean_test / test_seq_length
			#store losses
			loss_vec_test[i_epoch,kkt] = total_loss_test
			loss_vec_clean_test[i_epoch,kkt] = total_loss_clean_test
			pred_validity_vec_test[i_epoch,kkt] = np.argmax(pw_loss_test > err_thresh)*model_params['delta_t']
			pred_validity_vec_clean_test[i_epoch,kkt] = np.argmax(pw_loss_clean_test > err_thresh)*model_params['delta_t']


			# compute KL divergence between long predictions and whole test set:
			if compute_kl:
				kl_vec_inv_test[i_epoch,kkt,:] = kl4dummies(
								f_unNormalize_Y(normz_info, y_noisy_test),
								f_unNormalize_Y(normz_info, long_predictions))
				kl_vec_inv_clean_test[i_epoch,kkt,:] = kl4dummies(
								f_unNormalize_Y(normz_info, y_clean_test),
								f_unNormalize_Y(normz_info, long_predictions))
		# print updates every 10 iterations or in 10% incrememnts
		if  i_epoch % int( max(2, np.ceil(n_epochs/10)) ) == 0:
			print("Epoch {0}\nTotal run-time = {1}\nTraining Loss = {2}\nTesting Loss = {3}".format(
						i_epoch,
						str(timedelta(seconds=time()-t0)),
						total_loss_train.data.item(),
						np.mean(loss_vec_test[i_epoch,:])))

		if  (i_epoch==(n_epochs-1)) or (save_iterEpochs and (i_epoch % int( max(2, np.ceil(n_epochs/10)) ) == 0)):
			 # plot predictions vs truth
			# fig, (ax1, ax3) = plt.subplots(1, 2)
			fig, (ax_list) = plt.subplots(len(plot_state_indices),1)
			if not isinstance(ax_list,np.ndarray):
				ax_list = [ax_list]
			# first run and plot training fits
			hidden_state = torch.zeros((hidden_size, 1)).type(dtype)
			# init.normal_(hidden_state,0.0,0.1)
			predictions = np.zeros([train_seq_length, output_size])
			pred = output_train[0,:,None]
			predictions[0,:] = np.squeeze(output_train[0,:,None])
			saved_hidden_states = np.zeros([train_seq_length, hidden_size])
			saved_hidden_states[0,:] = hidden_state.data.numpy().ravel()
			solver_failed = False
			for i in range(train_seq_length-1):
				(pred, hidden_state, solver_failed) = forward(output_train[i,:,None], hidden_state, A,B,C,a,b, normz_info, model, model_params, model_output=model_pred[i], solver_failed=solver_failed)
				# (pred, hidden_state) = forward(pred, hidden_state, A,B,C,a,b, normz_info, model, model_params)
				# hidden_state = hidden_state
				saved_hidden_states[i+1,:] = hidden_state.data.numpy().ravel()
				predictions[i+1,:] = pred.data.numpy().ravel()

			y_noisy_train_raw = f_unNormalize_Y(normz_info,y_noisy_train)
			y_clean_train_raw = f_unNormalize_Y(normz_info,y_clean_train)
			predictions_raw = f_unNormalize_Y(normz_info,predictions)
			# y_clean_test_raw = y_clean_test
			# y_noisy_train_raw = y_noisy_train
			# y_clean_train_raw = y_clean_train
			# predictions_raw = predictions
			for kk in range(len(ax_list)):
				ax1 = ax_list[kk]
				t_plot = np.arange(0,round(len(y_noisy_train[:n_plttrain,plot_state_indices[kk]])*model_params['delta_t'],8),model_params['delta_t'])
				ax1.scatter(t_plot, y_noisy_train_raw[:n_plttrain,plot_state_indices[kk]], color='red', s=10, alpha=0.3, label='noisy data')
				ax1.plot(t_plot, y_clean_train_raw[:n_plttrain,plot_state_indices[kk]], color='red', label='clean data')
				ax1.plot(t_plot, predictions_raw[:n_plttrain,plot_state_indices[kk]], color='black', label='NN fit')
				ax1.set_xlabel('time')
				ax1.set_ylabel(model_params['state_names'][plot_state_indices[kk]] + '(t)', color='red')
				ax1.tick_params(axis='y', labelcolor='red')

			ax_list[0].legend()

			fig.suptitle('Training Fit')
			fig.savefig(fname=output_dir+'/rnn_train_fit_ode_iterEpochs'+str(i_epoch))
			plt.close(fig)

			# plot dynamics of hidden state over training set
			n_hidden_plots = min(10, hidden_size)
			fig, (ax_list) = plt.subplots(n_hidden_plots,1)
			if not isinstance(ax_list,np.ndarray):
				ax_list = [ax_list]
			for kk in range(len(ax_list)):
				ax1 = ax_list[kk]
				t_plot = np.arange(0,round(len(saved_hidden_states[:n_plttrain,kk])*model_params['delta_t'],8),model_params['delta_t'])
				ax1.plot(t_plot, saved_hidden_states[:n_plttrain,kk], color='red', label='clean data')
				ax1.set_xlabel('time')
				ax1.set_ylabel('h_{}'.format(kk), color='red')
				ax1.tick_params(axis='y', labelcolor='red')

			ax_list[0].legend()

			fig.suptitle('Hidden State Dynamics')
			fig.savefig(fname=output_dir+'/rnn_train_hidden_states_iterEpochs'+str(i_epoch))
			plt.close(fig)


			# now, loop over testing trajectories
			for kkt in range(n_test_sets):
				y_clean_test_raw = f_unNormalize_Y(normz_info,y_clean_test[kkt,:,:])
				fig, (ax_list) = plt.subplots(len(plot_state_indices),1)
				if not isinstance(ax_list,np.ndarray):
					ax_list = [ax_list]

				# NOW, show testing fit
				if synch_length > 1:
					# first, synchronize the hidden state
					hidden_state = torch.zeros((hidden_size, 1)).type(dtype)
					solver_failed = False
					for i in range(synch_length-1):
						(pred, hidden_state, solver_failed) = forward(test_synch_noisy[kkt,i,:,None], hidden_state, A,B,C,a,b , normz_info, model, model_params, solver_failed=solver_failed)
					pred = test_synch_noisy[kkt,-1,:,None]
				else:
					pred = output_train[-1,:,None]

				# hidden_state = Variable(torch.zeros((hidden_size, 1)).type(dtype), requires_grad=False)
				predictions = np.zeros([test_seq_length, output_size])
				# pred = output_train[-1,:,None]
				saved_hidden_states = np.zeros([test_seq_length, hidden_size])
				solver_failed = False
				for i in range(test_seq_length):
					(pred, hidden_state, solver_failed) = forward(pred, hidden_state, A,B,C,a,b, normz_info, model, model_params, solver_failed=solver_failed)
					# hidden_state = hidden_state
					predictions[i,:] = pred.data.numpy().ravel()
					saved_hidden_states[i,:] = hidden_state.data.numpy().ravel()

				predictions_raw = f_unNormalize_Y(normz_info,predictions)
				for kk in range(len(ax_list)):
					ax3 = ax_list[kk]
					t_plot = np.arange(0,len(y_clean_test[kkt,:n_plttest,plot_state_indices[kk]])*model_params['delta_t'],model_params['delta_t'])
					# ax3.scatter(np.arange(len(y_noisy_test)), y_noisy_test, color='red', s=10, alpha=0.3, label='noisy data')
					ax3.plot(t_plot, y_clean_test_raw[:n_plttest,plot_state_indices[kk]], color='red', label='clean data')
					ax3.plot(t_plot, predictions_raw[:n_plttest,plot_state_indices[kk]], color='black', label='NN fit')
					ax3.set_xlabel('time')
					ax3.set_ylabel(model_params['state_names'][plot_state_indices[kk]] +'(t)', color='red')
					ax3.tick_params(axis='y', labelcolor='red')

				ax_list[0].legend()

				fig.suptitle('RNN TEST fit to ODE simulation--' + str(i_epoch) + 'training epochs')
				fig.savefig(fname=output_dir+'/rnn_test_fit_ode_iterEpochs{0}_test{1}'.format(i_epoch,kkt))
				plt.close(fig)

				# plot dynamics of hidden state over TESTING set
				n_hidden_plots = min(10, hidden_size)
				fig, (ax_list) = plt.subplots(n_hidden_plots,1)
				if not isinstance(ax_list,np.ndarray):
					ax_list = [ax_list]
				for kk in range(len(ax_list)):
					ax1 = ax_list[kk]
					t_plot = np.arange(0,round(len(saved_hidden_states[:n_plttest,kk])*model_params['delta_t'],8),model_params['delta_t'])
					ax1.plot(t_plot, saved_hidden_states[:n_plttest,kk], color='red', label='clean data')
					ax1.set_xlabel('time')
					ax1.set_ylabel('h_{}'.format(kk), color='red')
					ax1.tick_params(axis='y', labelcolor='red')

				ax_list[0].legend()

				fig.suptitle('Hidden State Dynamics')
				fig.savefig(fname=output_dir+'/rnn_test_hidden_states_ode_iterEpochs{0}_test{1}'.format(i_epoch,kkt))
				plt.close(fig)

				# plot KDE of test data vs predictiosn
				fig, (ax_list) = plt.subplots(len(plot_state_indices),1)
				if not isinstance(ax_list,np.ndarray):
					ax_list = [ax_list]

				for kk in range(len(ax_list)):
					ax1 = ax_list[kk]
					pk = plot_state_indices[kk]
					try:
						x_grid = np.linspace(min(y_clean_test_raw[:,pk]), max(y_clean_test_raw[:,pk]), 1000)
						ax1.plot(x_grid, kde_func(y_clean_test_raw[:,pk], x_grid), label='clean data')
					except:
						pass
					try:
						x_grid = np.linspace(min(predictions_raw[:,pk]), max(predictions_raw[:,pk]), 1000)
						ax1.plot(x_grid, kde_func(predictions_raw[:,pk], x_grid), label='RNN fit')
					except:
						pass
					ax1.set_xlabel(model_params['state_names'][pk])

				ax_list[0].legend()

				fig.suptitle('Predictions of Invariant Density')
				fig.savefig(fname=output_dir+'/rnn_test_invDensity_iterEpochs{0}_test{1}'.format(i_epoch,kkt))
				plt.close(fig)


	## save loss_vec
	if n_epochs == 1:
		np.savetxt(output_dir+'/loss_vec_train.txt',running_epoch_loss_train)
		np.savetxt(output_dir+'/loss_vec_clean_train.txt',running_epoch_loss_clean_train)
		np.savetxt(output_dir+'/loss_vec_test.txt',running_epoch_loss_test)
		np.savetxt(output_dir+'/loss_vec_clean_test.txt',running_epoch_loss_clean_test)
	else:
		np.savetxt(output_dir+'/loss_vec_train.txt',loss_vec_train)
		np.savetxt(output_dir+'/loss_vec_clean_train.txt',loss_vec_clean_train)
		np.savetxt(output_dir+'/loss_vec_test.txt',loss_vec_test)
		np.savetxt(output_dir+'/loss_vec_clean_test.txt',loss_vec_clean_test)
		np.savetxt(output_dir+'/prediction_validity_time_test.txt',pred_validity_vec_test)
		np.savetxt(output_dir+'/prediction_validity_time_clean_test.txt',pred_validity_vec_clean_test)
		# np.savetxt(output_dir+'/kl_vec_inv_test.txt',kl_vec_inv_test)
		# np.savetxt(output_dir+'/kl_vec_inv_clean_test.txt',kl_vec_inv_clean_test)
		np.save(output_dir+'/kl_vec_inv_test.npy',kl_vec_inv_test)
		np.save(output_dir+'/kl_vec_inv_clean_test.npy',kl_vec_inv_clean_test)

	np.savetxt(output_dir+'/A_mat.txt',A.detach().numpy())
	np.savetxt(output_dir+'/B_mat.txt',B.detach().numpy())
	np.savetxt(output_dir+'/C_mat.txt',C.detach().numpy())
	np.savetxt(output_dir+'/a_vec.txt',a.detach().numpy())
	np.savetxt(output_dir+'/b_vec.txt',b.detach().numpy())
	# print("W1:",w1)
	# print("W2:",w2)
	# print("b:",b)
	# print("c:",c)
	# print("v:",v)

	# plot parameter convergence
	# if keep_param_history:
	fig, my_axes = plt.subplots(2, 3, sharex=True, figsize=[8,6])

	x_vals = np.linspace(0,n_epochs,len(A_history))
	my_axes[0,0].plot(x_vals, A_history)
	# my_axes[0,0].plot(x_vals, A_history_running)
	# my_axes[0,0].plot(x_vals, np.linalg.norm(A.detach().numpy() - A_history, ord='fro', axis=(1,2)))
	my_axes[0,0].set_title('A')
	my_axes[0,0].set_xlabel('Epochs')

	# my_axes[0,1].plot(x_vals, np.linalg.norm(B.detach().numpy() - B_history, ord='fro', axis=(1,2)))
	my_axes[0,1].plot(x_vals, B_history)
	# my_axes[0,1].plot(x_vals, B_history_running)
	my_axes[0,1].set_title('B')
	my_axes[0,1].set_xlabel('Epochs')

	# my_axes[1,0].plot(x_vals, np.linalg.norm(C.detach().numpy() - C_history, ord='fro', axis=(1,2)))
	my_axes[1,0].plot(x_vals, C_history)
	# my_axes[1,0].plot(x_vals, C_history_running)
	my_axes[1,0].set_title('C')
	my_axes[1,0].set_xlabel('Epochs')

	# my_axes[1,1].plot(x_vals, np.linalg.norm(a.detach().numpy() - a_history, ord='fro', axis=(1,2)))
	my_axes[1,1].plot(x_vals, a_history)
	# my_axes[1,1].plot(x_vals, a_history_running)
	my_axes[1,1].set_title('a')
	my_axes[1,1].set_xlabel('Epochs')

	# my_axes[1,2].plot(x_vals, np.linalg.norm(b.detach().numpy() - b_history, ord='fro', axis=(1,2)))
	my_axes[1,2].plot(x_vals, b_history)
	# my_axes[1,2].plot(x_vals, b_history_running)
	my_axes[1,2].set_title('b')
	my_axes[1,2].set_xlabel('Epochs')

	fig.suptitle("Parameter convergence")
	fig.savefig(fname=output_dir+'/rnn_parameter_convergence')
	plt.close(fig)

	## now, inspect the quality of the learned model
	# plot predictions vs truth
	# fig, (ax1, ax3) = plt.subplots(1, 2)
	fig, (ax_list) = plt.subplots(len(plot_state_indices),1)
	if not isinstance(ax_list,np.ndarray):
		ax_list = [ax_list]

	# first run and plot training fits
	hidden_state = torch.zeros((hidden_size, 1)).type(dtype)
	init.normal_(hidden_state,0.0,0.1)
	predictions = np.zeros([train_seq_length, output_size])
	pred = output_train[0,:,None]
	predictions[0,:] = np.squeeze(output_train[0,:,None])
	solver_failed = False
	for i in range(train_seq_length-1):
		(pred, hidden_state, solver_failed) = forward(output_train[i,:,None], hidden_state, A,B,C,a,b, normz_info, model, model_params, model_output=model_pred[i], solver_failed=solver_failed)
		# (pred, hidden_state) = forward(pred, hidden_state, A,B,C,a,b, normz_info, model, model_params)
		# hidden_state = hidden_state
		predictions[i+1,:] = pred.data.numpy().ravel()

	predictions_raw = f_unNormalize_Y(normz_info,predictions)
	for kk in range(len(ax_list)):
		ax1 = ax_list[kk]
		t_plot = np.arange(0,round(len(y_noisy_train_raw[:n_plttrain,plot_state_indices[kk]])*model_params['delta_t'],8),model_params['delta_t'])
		ax1.scatter(t_plot, y_noisy_train_raw[:n_plttrain,plot_state_indices[kk]], color='red', s=10, alpha=0.3, label='noisy data')
		ax1.plot(t_plot, y_clean_train_raw[:n_plttrain,plot_state_indices[kk]], color='red', label='clean data')
		ax1.plot(t_plot, predictions_raw[:n_plttrain,plot_state_indices[kk]], color='black', label='NN fit')
		ax1.set_xlabel('time')
		ax1.set_ylabel(model_params['state_names'][plot_state_indices[kk]] +'(t)', color='red')
		ax1.tick_params(axis='y', labelcolor='red')
		# ax1.set_title('Training Fit')

	ax_list[0].legend()
	fig.suptitle('RNN TRAIN fit to ODE simulation')
	fig.savefig(fname=output_dir+'/rnn_fit_ode_TRAIN')
	plt.close(fig)


	# NOW, show final testing fit
	for kkt in range(n_test_sets):
		y_clean_test_raw = f_unNormalize_Y(normz_info,y_clean_test[kkt,:,:])
		fig, (ax_list) = plt.subplots(len(plot_state_indices),1)
		if not isinstance(ax_list,np.ndarray):
			ax_list = [ax_list]

		# NOW, show testing fit
		if synch_length > 1:
			# first, synchronize the hidden state
			hidden_state = torch.zeros((hidden_size, 1)).type(dtype)
			solver_failed = False
			for i in range(synch_length-1):
				(pred, hidden_state, solver_failed) = forward(test_synch_noisy[kkt,i,:,None], hidden_state, A,B,C,a,b , normz_info, model, model_params, solver_failed=solver_failed)
			pred = test_synch_noisy[kkt,-1,:,None]
		else:
			pred = output_train[-1,:,None]

		# hidden_state = Variable(torch.zeros((hidden_size, 1)).type(dtype), requires_grad=False)
		predictions = np.zeros([test_seq_length, output_size])
		# pred = output_train[-1,:,None]
		saved_hidden_states = np.zeros([test_seq_length, hidden_size])
		solver_failed = False
		for i in range(test_seq_length):
			(pred, hidden_state, solver_failed) = forward(pred, hidden_state, A,B,C,a,b, normz_info, model, model_params, solver_failed=solver_failed)
			# hidden_state = hidden_state
			predictions[i,:] = pred.data.numpy().ravel()
			saved_hidden_states[i,:] = hidden_state.data.numpy().ravel()

		predictions_raw = f_unNormalize_Y(normz_info,predictions)
		for kk in range(len(ax_list)):
			ax3 = ax_list[kk]
			t_plot = np.arange(0,len(y_clean_test[kkt,:n_plttest,plot_state_indices[kk]])*model_params['delta_t'],model_params['delta_t'])
			# ax3.scatter(np.arange(len(y_noisy_test)), y_noisy_test, color='red', s=10, alpha=0.3, label='noisy data')
			ax3.plot(t_plot, y_clean_test_raw[:n_plttest,plot_state_indices[kk]], color='red', label='clean data')
			ax3.plot(t_plot, predictions_raw[:n_plttest,plot_state_indices[kk]], color='black', label='NN fit')
			ax3.set_xlabel('time')
			ax3.set_ylabel(model_params['state_names'][plot_state_indices[kk]] +'(t)', color='red')
			ax3.tick_params(axis='y', labelcolor='red')

		ax_list[0].legend()

		fig.suptitle('RNN TEST fit to ODE simulation')
		fig.savefig(fname=output_dir+'/rnn_fit_ode_TEST_{0}'.format(kkt))
		plt.close(fig)

		# plot KDE of test data vs predictiosn
		fig, (ax_list) = plt.subplots(len(plot_state_indices),1)
		if not isinstance(ax_list,np.ndarray):
			ax_list = [ax_list]

		for kk in range(len(ax_list)):
			ax1 = ax_list[kk]
			pk = plot_state_indices[kk]
			try:
				x_grid = np.linspace(min(y_clean_test_raw[:,pk]), max(y_clean_test_raw[:,pk]), 1000)
				ax1.plot(x_grid, kde_func(y_clean_test_raw[:,pk], x_grid), label='clean data')
			except:
				pass
			try:
				x_grid = np.linspace(min(predictions_raw[:,pk]), max(predictions_raw[:,pk]), 1000)
				ax1.plot(x_grid, kde_func(predictions_raw[:,pk], x_grid), label='RNN fit')
			except:
				pass
			ax1.set_xlabel(model_params['state_names'][pk])

		ax_list[0].legend()

		fig.suptitle('Predictions of Invariant Density')
		fig.savefig(fname=output_dir+'/rnn_invDensity_TEST_{0}'.format(kkt))
		plt.close(fig)


	# plot Train/Test curve
	x_test = pd.DataFrame(np.loadtxt(output_dir+"/loss_vec_clean_test.txt"))
	n_vals = n_epochs
	epoch_vec = np.arange(0,n_vals)
	max_exp = int(np.floor(np.log10(n_vals)))
	win_list = [1] + list(10**np.arange(1,max_exp))
	for win in win_list:
		fig, ax_vec = plt.subplots(nrows=2, ncols=2,
			figsize = [10, 10],
			sharey=False, sharex=False)

		ax1 = ax_vec[0,0]
		ax2 = ax_vec[0,1]
		ax3 = ax_vec[1,0]
		ax4 = ax_vec[1,1]

		x_train = pd.DataFrame(np.loadtxt(output_dir+'/loss_vec_train.txt'))
		x_test = pd.DataFrame(np.loadtxt(output_dir+"/loss_vec_clean_test.txt"))

		if n_epochs > 1:
			x_valid_test = pd.DataFrame(np.loadtxt(output_dir+"/prediction_validity_time_clean_test.txt"))
			x_kl_test = np.load(output_dir+"/kl_vec_inv_clean_test.npy")

		ax1.plot(x_train.rolling(win).mean())
		ax2.errorbar(x=epoch_vec,y=x_test.median(axis=1).rolling(win).mean(), yerr=x_test.std(axis=1), label='RNN')
		if n_epochs > 1:
			ax3.errorbar(x=epoch_vec,y=x_valid_test.median(axis=1).rolling(win).mean(), yerr=x_valid_test.std(axis=1), label='RNN')
			for kk in plot_state_indices:
				ax4.errorbar(x=epoch_vec,y=pd.DataFrame(np.median(x_kl_test[:,:,kk],axis=1)).rolling(win).mean().loc[:,0], yerr=np.std(x_kl_test[:,:,kk],axis=1), label='RNN')

		for gp_style in style_list:
			gp_nm = 'GPR {0}'.format(gp_style)
			gpr_valid_test = np.loadtxt(output_dir+'/GPR{0}_prediction_validity_time_clean_test.txt'.format(gp_style))
			gpr_test = np.loadtxt(output_dir+'/GPR{0}_loss_vec_clean_test.txt'.format(gp_style))
			ax2.errorbar(x=epoch_vec,y=[np.median(gpr_test)]*len(epoch_vec), yerr=[np.std(gpr_test)]*len(epoch_vec),label=gp_nm)
			ax3.errorbar(x=epoch_vec,y=[np.median(gpr_valid_test)]*len(epoch_vec), yerr=[np.std(gpr_valid_test)]*len(epoch_vec),label=gp_nm)

		ax2.legend()
		ax3.legend()

		# x = np.loadtxt(d+"/loss_vec_test.txt")
		# ax3.plot(x, label=d_label)
		ax4.set_xlabel('Epochs')
		ax3.set_xlabel('Epochs')
		ax2.set_xlabel('Epochs')
		ax1.set_xlabel('Epochs')

		ax1.set_ylabel('Error')
		ax1.set_title('Train Error')
		# ax1.legend(fontsize=6, handlelength=2, loc='upper right')
		ax2.set_ylabel('Error')
		ax2.set_title('Test Error (predicting clean data)')
		ax3.set_ylabel('Valid Time')
		ax3.set_title('Test Validity Time')
		ax4.set_title('KL-divergence')
		# ax3.set_xlabel('Epochs')
		# ax3.set_ylabel('Error')
		# ax3.set_title('Test Error (on noisy data)')
		# fig.suptitle("Comparison of training efficacy (trained on noisy data)")
		fig.savefig(fname=output_dir+'/TrainTest_win'+str(win))
		# plot in log scale
		ax1.set_yscale('log')
		ax2.set_yscale('log')
		ax3.set_yscale('log')
		ax4.set_yscale('log')
		ax1.set_ylabel('log Error')
		ax2.set_ylabel('log Error')
		ax3.set_ylabel('Valid Time')
		fig.savefig(fname=output_dir+'/TrainTest_log'+'_win'+str(win))
		plt.close(fig)


def train_RNN(forward,
			y_clean_train, y_noisy_train, x_train,
			y_clean_test, y_noisy_test, x_test,
			model_params, hidden_size=6, n_epochs=100, lr=0.05,
			output_dir='.', normz_info=None, model=None,
			trivial_init=False, drive_system=True):

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	print('Starting RNN training for: ', output_dir)

	if drive_system:
		x_train = torch.FloatTensor(x_train)
		x_test = torch.FloatTensor(x_test)
		input_size = x_train.shape[0]

	output_train = torch.FloatTensor(y_noisy_train)
	output_clean_train = torch.FloatTensor(y_clean_train)
	output_test = torch.FloatTensor(y_noisy_test)
	output_clean_test = torch.FloatTensor(y_clean_test)

	dtype = torch.FloatTensor
	output_size = output_train.shape[1]
	train_seq_length = output_train.size(0)
	test_seq_length = output_test.size(0)

	# first, SHOW that a simple mechRNN can fit the data perfectly
	# now, TRAIN to fit the output from the previous model
	if drive_system:
		w2 = torch.zeros(hidden_size, input_size).type(dtype)
	w1 = torch.zeros(hidden_size, hidden_size).type(dtype)
	w1[0,0] = 1.
	b = torch.zeros(hidden_size, 1).type(dtype)
	c = torch.zeros(output_size, 1).type(dtype)
	v = torch.zeros(output_size, hidden_size).type(dtype)
	v[0] = 1.
	hidden_state = torch.zeros((hidden_size, 1)).type(dtype)
	predictions = []
	# yb_normalized = (yb - YMIN)/(YMAX - YMIN)
	# initializing y0 of hidden state to the true initial condition from the clean signal

	hidden_state[0] = float(y_clean_test[0])
	for i in range(test_seq_length):
		(pred, hidden_state) = forward(x_test[:,i:i+1], hidden_state, w1, w2, b, c, v, normz_info, model, model_params)
		hidden_state = hidden_state
		predictions.append(pred.data.numpy().ravel()[0])
	# plot predictions vs truth
	fig, ax1 = plt.subplots()

	ax1.scatter(np.arange(len(y_noisy_test)), y_noisy_test, color='red', s=10, alpha=0.3, label='noisy data')
	ax1.plot(y_clean_test, color='red', label='clean data')
	ax1.plot(predictions, ':' ,color='red', label='NN trivial fit')
	ax1.set_xlabel('time')
	ax1.set_ylabel('y(t)', color='red')
	ax1.tick_params(axis='y', labelcolor='red')

	if drive_system:
		ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
		color = 'tab:blue'
		ax2.set_ylabel('x(t)', color=color)  # we already handled the x-label with ax1
		ax2.plot((x_test[0,:].numpy()), ':', color=color, linestyle='--', label='input/driver data')
		ax2.tick_params(axis='y', labelcolor=color)

	fig.legend()
	fig.suptitle('RNN w/ just mechanism fit to ODE simulation TEST SET')
	fig.savefig(fname=output_dir+'/PERFECT_MechRnn_fit_ode')
	plt.close(fig)

	# Initilize parameters for training
	if drive_system:
		w2 = torch.FloatTensor(hidden_size, input_size).type(dtype)
	w1 = torch.FloatTensor(hidden_size, hidden_size).type(dtype)
	b = torch.FloatTensor(hidden_size, 1).type(dtype)
	c = torch.FloatTensor(output_size, 1).type(dtype)
	v = torch.FloatTensor(output_size, hidden_size).type(dtype)

	# trivial_init trains the mechRNN starting from parameters (specified above)
	# that trivialize the RNN to the forward ODE model
	# now, TRAIN to fit the output from the previous model
	if trivial_init:
		if drive_system:
			init.zeros_(w2)
		init.zeros_(w1)
		init.zeros_(b)
		init.zeros_(c)
		init.zeros_(v)
		w1[0,0] = 1.
		v[0] = 1.
	else:
		if drive_system:
			init.normal_(w2, 0.0, 0.1)
		init.normal_(w1, 0.0, 0.1)
		init.normal_(b, 0.0, 0.1)
		init.normal_(c, 0.0, 0.1)
		init.normal_(v, 0.0, 0.1)

	if drive_system:
		w2 = Variable(w2, requires_grad=True)
	w1 =  Variable(w1, requires_grad=True)
	b =  Variable(b, requires_grad=True)
	c =  Variable(c, requires_grad=True)
	v =  Variable(v, requires_grad=True)

	if drive_system:
		w2_history = np.zeros((n_epochs*train_seq_length,w2.shape[0],w2.shape[1]))
	w1_history = np.zeros((n_epochs*train_seq_length,w1.shape[0],w1.shape[1]))
	b_history = np.zeros((n_epochs*train_seq_length,b.shape[0],b.shape[1]))
	c_history = np.zeros((n_epochs*train_seq_length,c.shape[0],c.shape[1]))
	v_history = np.zeros((n_epochs*train_seq_length,v.shape[0],v.shape[1]))

	loss_vec_train = np.zeros(n_epochs)
	loss_vec_clean_train = np.zeros(n_epochs)
	loss_vec_test = np.zeros(n_epochs)
	loss_vec_clean_test = np.zeros(n_epochs)
	cc = -1 # iteration counter for saving weight updates
	for i_epoch in range(n_epochs):
		total_loss_train = 0
		total_loss_clean_train = 0
		#init.normal_(hidden_state, 0.0, 1)
		#hidden_state = Variable(hidden_state, requires_grad=True)
		hidden_state = Variable(torch.zeros((hidden_size, 1)).type(dtype), requires_grad=False)
		for j in range(train_seq_length):
			cc += 1
			target = output_train[j:(j+1)]
			target_clean = output_clean_train[j:(j+1)]
			if drive_system:
				(pred, hidden_state) = forward(x_train[:,j:j+1], hidden_state, w1, w2, b, c, v, normz_info, model, model_params)
			else:
				(pred, hidden_state) = forward(hidden_state, w1, b, c, v, normz_info, model, model_params)
			loss = (pred.squeeze() - target.squeeze()).pow(2).sum()
			total_loss_train += loss
			total_loss_clean_train += (pred.squeeze() - target_clean.squeeze()).pow(2).sum()
			loss.backward()

			if drive_system:
				w2.data -= lr * w2.grad.data
			w1.data -= lr * w1.grad.data
			b.data -= lr * b.grad.data
			c.data -= lr * c.grad.data
			v.data -= lr * v.grad.data

			if drive_system:
				w2.grad.data.zero_()
			w1.grad.data.zero_()
			b.grad.data.zero_()
			c.grad.data.zero_()
			v.grad.data.zero_()

			hidden_state = hidden_state.detach()

			# save updated parameters
			if drive_system:
				w2_history[cc,:] = w2.detach().numpy()
			w1_history[cc,:] = w1.detach().numpy()
			b_history[cc,:] = b.detach().numpy()
			c_history[cc,:] = c.detach().numpy()
			v_history[cc,:] = v.detach().numpy()

		#normalize losses
		total_loss_train = total_loss_train / train_seq_length
		total_loss_clean_train = total_loss_clean_train / train_seq_length
		#store losses
		loss_vec_train[i_epoch] = total_loss_train
		loss_vec_clean_train[i_epoch] = total_loss_clean_train

		total_loss_test = 0
		total_loss_clean_test = 0
		hidden_state = Variable(torch.zeros((hidden_size, 1)).type(dtype), requires_grad=False)
		for j in range(test_seq_length):
			target = output_test[j:(j+1)]
			target_clean = output_clean_test[j:(j+1)]
			if drive_system:
				(pred, hidden_state) = forward(x_test[:,j:j+1], hidden_state, w1, w2, b, c, v, normz_info, model, model_params)
			else:
				(pred, hidden_state) = forward(hidden_state, w1, b, c, v, normz_info, model, model_params)
			total_loss_test += (pred.squeeze() - target.squeeze()).pow(2).sum()
			total_loss_clean_test += (pred.squeeze() - target_clean.squeeze()).pow(2).sum()

			hidden_state = hidden_state.detach()
		#normalize losses
		total_loss_test = total_loss_test / test_seq_length
		total_loss_clean_test = total_loss_clean_test / test_seq_length
		#store losses
		loss_vec_test[i_epoch] = total_loss_test
		loss_vec_clean_test[i_epoch] = total_loss_clean_test

		# print updates every 10 iterations or in 10% incrememnts
		if i_epoch % int( max(10, np.ceil(n_epochs/10)) ) == 0:
			print("Epoch: {}\nTraining Loss = {}\nTesting Loss = {}".format(
						i_epoch,
						total_loss_train.data.item(),
						total_loss_test.data.item()))
				 # plot predictions vs truth
			fig, (ax1, ax3) = plt.subplots(1, 2)

			# first run and plot training fits
			hidden_state = Variable(torch.zeros((hidden_size, 1)).type(dtype), requires_grad=False)
			predictions = []
			for i in range(train_seq_length):
				if drive_system:
					(pred, hidden_state) = forward(x_train[:,i:i+1], hidden_state, w1, w2, b, c, v, normz_info, model, model_params)
				else:
					(pred, hidden_state) = forward(hidden_state, w1, b, c, v, normz_info, model, model_params)
				hidden_state = hidden_state
				predictions.append(pred.data.numpy().ravel()[0])

			ax1.scatter(np.arange(len(y_noisy_train[:,0])), y_noisy_train[:,0], color='red', s=10, alpha=0.3, label='noisy data')
			ax1.plot(y_clean_train[:,0], color='red', label='clean data')
			ax1.plot(predictions, color='black', label='NN fit')
			ax1.set_xlabel('time')
			ax1.set_ylabel('y(t)', color='red')
			ax1.tick_params(axis='y', labelcolor='red')


			if drive_system:
				ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
				color = 'tab:blue'
				ax2.set_ylabel('x(t)', color=color)  # we already handled the x-label with ax1
				ax2.plot((x_train[0,:].numpy()), ':', color=color, linestyle='--', label='driver/input data')
				ax2.tick_params(axis='y', labelcolor=color)

			ax1.set_title('Training Fit')

			# NOW, show testing fit
			hidden_state = Variable(torch.zeros((hidden_size, 1)).type(dtype), requires_grad=False)
			predictions = []
			for i in range(test_seq_length):
				if drive_system:
					(pred, hidden_state) = forward(x_test[:,i:i+1], hidden_state, w1, w2, b, c, v, normz_info, model, model_params)
				else:
					(pred, hidden_state) = forward(hidden_state, w1, b, c, v, normz_info, model, model_params)
				hidden_state = hidden_state
				predictions.append(pred.data.numpy().ravel()[0])

			# ax3.scatter(np.arange(len(y_noisy_test)), y_noisy_test, color='red', s=10, alpha=0.3, label='noisy data')
			ax3.plot(y_clean_test[:,0], color='red', label='clean data')
			ax3.plot(predictions, color='black', label='NN fit')
			ax3.set_xlabel('time')
			ax3.set_ylabel('y(t)', color='red')
			ax3.tick_params(axis='y', labelcolor='red')

			if drive_system:
				ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
				color = 'tab:blue'
				ax4.set_ylabel('x(t)', color=color)  # we already handled the x-label with ax1
				ax4.plot((x_test[0,:].numpy()), ':', color=color, linestyle='--', label='driver/input data')
				ax4.tick_params(axis='y', labelcolor=color)

			ax3.set_title('Testing Fit')

			ax3.legend()
			ax2.legend()

			fig.suptitle('RNN fit to ODE simulation--' + str(i_epoch) + 'training epochs')
			fig.savefig(fname=output_dir+'/rnn_fit_ode_iterEpochs'+str(i_epoch))
			plt.close(fig)

			# Plot parameter convergence


	## save loss_vec
	np.savetxt(output_dir+'/loss_vec_train.txt',loss_vec_train)
	np.savetxt(output_dir+'/loss_vec_clean_train.txt',loss_vec_clean_train)
	np.savetxt(output_dir+'/loss_vec_test.txt',loss_vec_test)
	np.savetxt(output_dir+'/loss_vec_clean_test.txt',loss_vec_clean_test)
	np.savetxt(output_dir+'/w1.txt',w1.detach().numpy())
	if drive_system:
		np.savetxt(output_dir+'/w2.txt',w2.detach().numpy())
	np.savetxt(output_dir+'/b.txt',b.detach().numpy())
	np.savetxt(output_dir+'/c.txt',c.detach().numpy())
	np.savetxt(output_dir+'/v.txt',v.detach().numpy())

	# print("W1:",w1)
	# print("W2:",w2)
	# print("b:",b)
	# print("c:",c)
	# print("v:",v)

	# plot parameter convergence
	fig, my_axes = plt.subplots(2, 3, sharex=True, figsize=[8,6])

	x_vals = np.linspace(0,n_epochs,len(w1_history))
	my_axes[0,0].plot(x_vals, np.linalg.norm(w1.detach().numpy() - w1_history, ord='fro', axis=(1,2)))
	my_axes[0,0].set_title('W_1')
	my_axes[0,0].set_xlabel('Epochs')

	if drive_system:
		my_axes[0,1].plot(x_vals, np.linalg.norm(w2.detach().numpy() - w2_history, ord='fro', axis=(1,2)))
		my_axes[0,1].set_title('W_2')
		my_axes[0,1].set_xlabel('Epochs')

	my_axes[1,0].plot(x_vals, np.linalg.norm(v.detach().numpy() - v_history, ord='fro', axis=(1,2)))
	my_axes[1,0].set_title('v')
	my_axes[1,0].set_xlabel('Epochs')

	my_axes[1,1].plot(x_vals, np.linalg.norm(b.detach().numpy() - b_history, ord='fro', axis=(1,2)))
	my_axes[1,1].set_title('b')
	my_axes[1,1].set_xlabel('Epochs')

	my_axes[1,2].plot(x_vals, np.linalg.norm(c.detach().numpy() - c_history, ord='fro', axis=(1,2)))
	my_axes[1,2].set_title('c')
	my_axes[1,2].set_xlabel('Epochs')


	fig.suptitle("Parameter convergence")
	fig.savefig(fname=output_dir+'/rnn_parameter_convergence')
	plt.close(fig)

	## now, inspect the quality of the learned model
	# plot predictions vs truth
	fig, (ax1, ax3) = plt.subplots(1, 2)

	# first run and plot training fits
	hidden_state = Variable(torch.zeros((hidden_size, 1)).type(dtype), requires_grad=False)
	predictions = []
	for i in range(train_seq_length):
		if drive_system:
			(pred, hidden_state) = forward(x_train[:,i:i+1], hidden_state, w1, w2, b, c, v, normz_info, model, model_params)
		else:
			(pred, hidden_state) = forward(hidden_state, w1, b, c, v, normz_info, model, model_params)
		hidden_state = hidden_state
		predictions.append(pred.data.numpy().ravel()[0])

	ax1.scatter(np.arange(len(y_noisy_train[:,0])), y_noisy_train[:,0], color='red', s=10, alpha=0.3, label='noisy data')
	ax1.plot(y_clean_train[:,0], color='red', label='clean data')
	ax1.plot(predictions, color='black', label='NN fit')
	ax1.set_xlabel('time')
	ax1.set_ylabel('y(t)', color='red')
	ax1.tick_params(axis='y', labelcolor='red')

	if drive_system:
		ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
		color = 'tab:blue'
		ax2.set_ylabel('x(t)', color=color)  # we already handled the x-label with ax1
		ax2.plot((x_train[0,:].numpy()), ':', color=color, linestyle='--', label='driver/input data')
		ax2.tick_params(axis='y', labelcolor=color)

	ax1.set_title('Training Fit')

	# NOW, show testing fit
	hidden_state = Variable(torch.zeros((hidden_size, 1)).type(dtype), requires_grad=False)
	predictions = []
	for i in range(test_seq_length):
		if drive_system:
			(pred, hidden_state) = forward(x_test[:,i:i+1], hidden_state, w1, w2, b, c, v, normz_info, model, model_params)
		else:
			(pred, hidden_state) = forward(hidden_state, w1, b, c, v, normz_info, model, model_params)
		hidden_state = hidden_state
		predictions.append(pred.data.numpy().ravel()[0])

	# ax3.scatter(np.arange(len(y_noisy_test)), y_noisy_test, color='red', s=10, alpha=0.3, label='noisy data')
	ax3.plot(y_clean_test[:,0], color='red', label='clean data')
	ax3.plot(predictions[:,0], color='black', label='NN fit')
	ax3.set_xlabel('time')
	ax3.set_ylabel('y(t)', color='red')
	ax3.tick_params(axis='y', labelcolor='red')

	if drive_system:
		ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis
		color = 'tab:blue'
		ax4.set_ylabel('x(t)', color=color)  # we already handled the x-label with ax1
		ax4.plot((x_test[0,:].numpy()), ':', color=color, linestyle='--', label='driver/input data')
		ax4.tick_params(axis='y', labelcolor=color)

	ax3.set_title('Testing Fit')

	ax3.legend()
	ax2.legend()

	fig.suptitle('RNN fit to ODE simulation')
	fig.savefig(fname=output_dir+'/rnn_fit_ode')
	plt.close(fig)


def compare_fits(my_dirs, output_fname="./training_comparisons", plot_state_indices=None):

	# first, get sizes of things...max window size is 10% of whole test set.
	d_label = my_dirs[0].split("/")[-1].rstrip('_noisy').rstrip('_clean')
	x_test = pd.DataFrame(np.loadtxt(my_dirs[0]+"/loss_vec_clean_test.txt"))
	n_vals = x_test.shape[0]
	epoch_vec = np.arange(0,n_vals)
	max_exp = int(np.floor(np.log10(n_vals)))
	win_list = [1] + list(10**np.arange(1,max_exp))

	try:
		many_epochs = True
		x_kl_test = np.load(my_dirs[0]+"/kl_vec_inv_clean_test.npy")
		if not plot_state_indices:
			plot_state_indices = np.arange(x_kl_test.shape[2])
	except:
		many_epochs = False


	for win in win_list:
		fig, ax_vec = plt.subplots(nrows=2, ncols=2,
			figsize = [10, 10],
			sharey=False, sharex=False)

		ax1 = ax_vec[0,0]
		ax2 = ax_vec[0,1]
		ax3 = ax_vec[1,0]
		ax4 = ax_vec[1,1]

		for d in my_dirs:
			d_label = d.split("/")[-1].rstrip('_noisy').rstrip('_clean')
			# if 'GPR' in d_label:
			# 	continue
			x_test = np.loadtxt(d+"/loss_vec_clean_test.txt",ndmin=2)
			if 'GPR' not in d_label:
				x_train = pd.DataFrame(np.loadtxt(d+"/loss_vec_train.txt"))
				x_test = pd.DataFrame(x_test)

			# get GPR fits
			# gpr1_valid_test = np.loadtxt(d+'/GPR1_prediction_validity_time_clean_test.txt')
			# gpr2_valid_test = np.loadtxt(d+'/GPR2_prediction_validity_time_clean_test.txt')
			# gpr1_test = np.loadtxt(d+'/GPR1_loss_vec_clean_test.txt')
			# gpr2_test = np.loadtxt(d+'/GPR2_loss_vec_clean_test.txt')

			if many_epochs:
				x_valid_test = np.loadtxt(d+"/prediction_validity_time_clean_test.txt",ndmin=2)
				if 'GPR' not in d_label:
					x_valid_test = pd.DataFrame(x_valid_test)
			# if win:
			if 'GPR' not in d_label:
				ax1.plot(x_train.rolling(win).mean(), label=d_label)
				ax2.errorbar(x=epoch_vec,y=x_test.median(axis=1).rolling(win).mean(), yerr=x_test.std(axis=1), label=d_label)
			else:
				ax2.errorbar(x=epoch_vec,y=[np.median(x_test)]*len(epoch_vec), yerr=[np.std(x_test)]*len(epoch_vec), label=d_label)
			if many_epochs:
				if 'GPR' not in d_label:
					ax3.errorbar(x=epoch_vec,y=x_valid_test.median(axis=1).rolling(win).mean(), yerr=x_valid_test.std(axis=1), label=d_label)
					x_kl_test = np.load(d+"/kl_vec_inv_clean_test.npy")
					for kk in plot_state_indices:
						ax4.errorbar(x=epoch_vec,y=pd.DataFrame(np.median(x_kl_test[:,:,kk],axis=1)).rolling(win).mean().loc[:,0], yerr=np.std(x_kl_test[:,:,kk],axis=1), label=d_label)
				else:
					ax3.errorbar(x=epoch_vec,y=[np.median(x_valid_test)]*len(epoch_vec), yerr=[np.std(x_valid_test)]*len(epoch_vec),label=d_label)
			# else:
			# 	if 'GPR' not in d_label:
			# 		ax1.plot(x_train, label=d_label)
			# 		ax2.errorbar(x=epoch_vec,y=x_test.median(axis=1), yerr=x_test.std(axis=1), label=d_label)
			# 	else:
			# 		ax2.errorbar(x=epoch_vec,y=[np.median(x_test)]*len(epoch_vec), yerr=[np.std(x_test)]*len(epoch_vec), label=d_label)
			# 	if many_epochs:
			# 		if 'GPR' not in d_label:
			# 			ax3.errorbar(x=epoch_vec,y=x_valid_test.median(axis=1), yerr=x_valid_test.std(axis=1), label=d_label)
			# 			x_kl_test = np.load(d+"/kl_vec_inv_clean_test.npy")
			# 			for kk in plot_state_indices:
			# 				ax4.errorbar(x=epoch_vec,y=np.median(x_kl_test[:,:,kk],axis=1), yerr=np.std(x_kl_test[:,:,kk],axis=1), label=d_label)
			# 		else:
			# 			ax3.errorbar(x=epoch_vec,y=[np.median(x_valid_test)]*len(epoch_vec), yerr=[np.std(x_valid_test)]*len(epoch_vec),label=d_label)

			# try:
			# 	gp_label = [x for x in d_label.split('_') if 'epsBad' in x][0]
			# 	ax2.plot([0,n_vals-1],[gpr1_test,gpr1_test],label='GPR 1 ' + gp_label)
			# 	ax2.plot([0,n_vals-1],[gpr2_test,gpr2_test],label='GPR 2 '+gp_label)
			# 	ax3.plot([0,n_vals-1],[gpr1_valid_test,gpr1_valid_test],label='GPR 1 ' + gp_label)
			# 	ax3.plot([0,n_vals-1],[gpr2_valid_test,gpr2_valid_test],label='GPR 2 ' + gp_label)
			# except:
			# 	pass

		ax2.legend(fontsize=6, handlelength=2, loc='upper right')
			# x = np.loadtxt(d+"/loss_vec_test.txt")
			# ax3.plot(x, label=d_label)
		ax4.set_xlabel('Epochs')
		ax3.set_xlabel('Epochs')
		ax2.set_xlabel('Epochs')
		ax1.set_xlabel('Epochs')

		ax1.set_ylabel('Error')
		ax1.set_title('Train Error')
		ax1.legend(fontsize=6, handlelength=2, loc='upper right')
		ax2.set_ylabel('Error')
		ax2.set_title('Test Error (predicting clean data)')
		ax3.set_ylabel('Valid Time')
		ax3.set_title('Test Validity Time')
		ax4.set_title('KL-divergence of Invariant Density Approx')

		# ax3.set_xlabel('Epochs')
		# ax3.set_ylabel('Error')
		# ax3.set_title('Test Error (on noisy data)')
		# fig.suptitle("Comparison of training efficacy (trained on noisy data)")
		fig.savefig(fname=output_fname+'_win'+str(win))
		# plot in log scale
		ax1.set_yscale('log')
		ax2.set_yscale('log')
		ax3.set_yscale('log')
		ax4.set_yscale('log')
		ax1.set_ylabel('log Error')
		ax2.set_ylabel('log Error')
		ax3.set_ylabel('Valid Time')
		fig.savefig(fname=output_fname+'_log'+'_win'+str(win))
		plt.close(fig)


def run_3DVAR(y_clean, y_noisy, eta, G_assim, delta_t,
		model, model_params, lr, output_dir,
		H_obs_lowfi=None, H_obs_hifi=None, noisy_hifi=False,
		inits=None, plot_state_indices=None,
		max_plot=None, learn_assim=False, eps=None, cheat=False, new_cheat=False, h=1e-3, lr_G=0.0005,
		G_update_interval=1, N_q_tries=1, n_epochs=1, eps_hifi=None,
		full_sequence=False, optimization=None, opt_surfaces_only=False,
		optim_full_state=True, random_nelder_inits=True, n_nelder_inits=1,
		max_nelder_sols=None):

	dtype = torch.FloatTensor

	H_obs_lowfi = torch.FloatTensor(H_obs_lowfi)
	# H_obs_hifi = torch.FloatTensor(H_obs_hifi)

	# y_clean = torch.FloatTensor(y_clean)
	# y_noisy = torch.FloatTensor(y_noisy)

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	if max_plot is None:
		max_plot = int(np.floor(100./model_params['delta_t']))

	n_plt = min(max_plot,y_clean.shape[0])
	n_plt_start = y_clean.shape[0] - n_plt

	if not plot_state_indices:
		plot_state_indices = np.arange(y_clean.shape[1])

	y_clean_lowfi = np.matmul(H_obs_lowfi.numpy(),y_clean.T).T
	y_noisy_lowfi = np.matmul(H_obs_lowfi.numpy(),y_noisy.T).T

	y_hifi = y_clean

	# y_hifi = np.matmul(H_obs_hifi.numpy(),y_clean.T).T
	# if noisy_hifi:
	# 	if eps_hifi is None:
	# 		eps_hifi = eps
	# 	# add decoupled noise of size epsilon to hifi measurements
	# 	y_hifi = y_hifi + eps_hifi*np.random.randn(y_hifi.shape[0],y_hifi.shape[1])

	# if noisy_hifi:
	# 	y_hifi = np.matmul(H_obs_hifi.numpy(),y_noisy.T).T
	# else:
	# 	y_hifi = np.matmul(H_obs_hifi.numpy(),y_clean.T).T

	n_iters = y_clean.shape[0]
	y_assim = np.zeros(y_clean.shape)
	y_predictions = np.zeros(y_clean.shape)


	if inits is None:
		inits = get_lorenz_inits(n=1).squeeze()
		# inits = y_noisy[0,:] + np.random.randn()
		# inits = np.array([-5, 0, 30]) + 20*np.random.randn()

	tspan = [0, 0.5*delta_t, delta_t]
	# initialize G_assim variable
	G_assim = torch.FloatTensor(G_assim)
	if learn_assim and not full_sequence:
		# G_assim = torch.zeros(y_clean.shape[1], y_clean_lowfi.shape[1]).type(dtype)
		# init.normal_(G_assim, 0.0, 0.1)
		G_assim = Variable(G_assim, requires_grad=True)


	def f_mk(G, m_pred_now, meas_lowfi_now, H_obs_lowfi=H_obs_lowfi):
		foo_assim = torch.mm( torch.eye(G.shape[0]) - torch.mm(G, H_obs_lowfi), m_pred_now.detach()) + torch.mm(G,meas_lowfi_now)
		return foo_assim

	def f_Lk_cheat_OLD(G, m_pred_now, meas_lowfi_now, meas_hifi_now, H_obs_lowfi=H_obs_lowfi):
		mk = f_mk(G, m_pred_now, meas_lowfi_now)
		Lk = ( torch.mm(H_obs_lowfi, mk).squeeze() -  meas_hifi_now.squeeze() ).pow(2).sum()
		return Lk

	def f_Lk_cheat(G, m_pred_now, meas_lowfi_now, meas_hifi_now):
		mk = f_mk(G, m_pred_now, meas_lowfi_now)
		Lk = ( mk.squeeze() -  meas_hifi_now.squeeze() ).pow(2).sum()
		return Lk

	def f_Lk(G, m_assim_prev2, meas_prev1, meas_now, H=H_obs_lowfi):
		sol = solve_ivp(fun=lambda t, y: model(y, t, *model_params['ode_params']), t_span=(tspan[0], tspan[-1]), y0=m_assim_prev2.T, method='RK45', t_eval=tspan)
		m_pred_prev1 = torch.FloatTensor(sol.y.T[-1,:,None])
		m_assim_prev1 = f_mk(G, m_pred_prev1, meas_prev1).detach().numpy().squeeze()

		sol = solve_ivp(fun=lambda t, y: model(y, t, *model_params['ode_params']), t_span=(tspan[0], tspan[-1]), y0=m_assim_prev1.T, method='RK45', t_eval=tspan)
		m_pred_now = torch.FloatTensor(sol.y.T[-1,:,None])
		Lk = ( torch.mm(H, m_pred_now) -  meas_now ).pow(2).sum()
		return Lk

	def f_Loss_Sum_OLD(G_input, n_iters=n_iters, use_inits=inits):
		loss = 0
		loss_cheat = 0
		y_assim = np.zeros(y_clean.shape)
		for i in range(n_iters):
			meas_now = torch.FloatTensor(y_noisy_lowfi[i,:,None])

			# make prediction using previous state estimate
			sol = solve_ivp(fun=lambda t, y: model(y, t, *model_params['ode_params']), t_span=(tspan[0], tspan[-1]), y0=use_inits.T, method='RK45', t_eval=tspan)
			m_pred_now = torch.FloatTensor(sol.y.T[-1,:,None])
			y_predictions[i,:] = m_pred_now.detach().numpy().squeeze()

			# Do the assimilation!
			m_assim_now = f_mk(G_input, m_pred_now, meas_now)
			# compute loss
			if i > 200:
				loss += ( torch.mm(H_obs_lowfi, m_pred_now).squeeze() -  meas_now.squeeze() ).pow(2).sum()
				loss_cheat += ( m_pred_now.squeeze() -  torch.FloatTensor(y_clean[i,:]).squeeze() ).pow(2).sum()
				# loss += ( torch.mm(H_obs_lowfi, m_assim_now).squeeze() -  meas_now.squeeze() ).pow(2).sum()
				# loss_cheat += ( m_assim_now.squeeze() -  torch.FloatTensor(y_clean[i,:]).squeeze() ).pow(2).sum()
			# loss += ( torch.mm(H_obs_lowfi, m_pred_now).squeeze() -  meas_now.squeeze() ).pow(2).sum()

			# set inits
			use_inits = m_assim_now.detach().numpy().squeeze()

			y_assim[i,:] = use_inits

		# pw_assim_errors = np.linalg.norm(y_assim - y_clean, axis=1, ord=2)**2
		return loss/n_iters

	def f_Loss_Sum(G_input, n_iters=n_iters, use_inits=inits):
		print('G=',G_input)
		if type(G_input) is not torch.Tensor:
			G_input = torch.FloatTensor(G_input[:,None])

		in_stationary = False
		total_loss = 0
		stationary_loss = np.inf
		y_assim = np.zeros(y_clean.shape)
		for i in range(n_iters):
			meas_now = torch.FloatTensor(y_noisy_lowfi[i,:,None])

			# make prediction using previous state estimate
			sol = solve_ivp(fun=lambda t, y: model(y, t, *model_params['ode_params']), t_span=(tspan[0], tspan[-1]), y0=use_inits.T, method='RK45', t_eval=tspan)
			m_pred_now = torch.FloatTensor(sol.y.T[-1,:,None])
			y_predictions[i,:] = m_pred_now.detach().numpy().squeeze()

			# Do the assimilation!
			m_assim_now = f_mk(G_input, m_pred_now, meas_now)

			# set inits
			use_inits = m_assim_now.detach().numpy().squeeze()

			# compute loss
			tmp_loss = ( torch.mm(H_obs_lowfi, m_pred_now).squeeze() -  meas_now.squeeze() ).pow(2).sum()
			total_loss += tmp_loss

			if not in_stationary and (tmp_loss<=eps):
				in_stationary = True
				stationary_loss = 0

			if in_stationary:
				stationary_loss += tmp_loss

			y_assim[i,:] = use_inits

			if any(abs(m_assim_now) > 1000):
				print('Model is blowing up! Abort!')
				print(m_assim_now)
				n_iters = i
				break

		# pdb.set_trace()
		if in_stationary:
			loss = stationary_loss
		else:
			loss = total_loss

		print('Loss=',loss/n_iters)
		# pw_assim_errors = np.linalg.norm(y_assim - y_clean, axis=1, ord=2)**2
		return loss/n_iters


	def f_Loss_PartialState(G_input, partial_traj, n_max, use_inits=inits):
		print('G=',G_input)
		if type(G_input) is not torch.Tensor:
			G_input = torch.FloatTensor(G_input[:,None])

		in_stationary = False
		total_loss = 0
		stationary_loss = np.inf
		y_assim = np.zeros(y_clean.shape)
		for i in range(n_max):
			meas_now = torch.FloatTensor(partial_traj[i,:,None])

			# make prediction using previous state estimate
			sol = solve_ivp(fun=lambda t, y: model(y, t, *model_params['ode_params']), t_span=(tspan[0], tspan[-1]), y0=use_inits.T, method='RK45', t_eval=tspan)
			m_pred_now = torch.FloatTensor(sol.y.T[-1,:,None])
			# y_predictions[i,:] = m_pred_now.detach().numpy().squeeze()

			# Do the assimilation!
			m_assim_now = f_mk(G_input, m_pred_now, meas_now)

			# set inits
			use_inits = m_assim_now.detach().numpy().squeeze()

			# compute loss
			tmp_loss = ( torch.mm(H_obs_lowfi, m_pred_now).squeeze() -  meas_now.squeeze() ).pow(2).sum()
			total_loss += tmp_loss

			if not in_stationary and (tmp_loss<=eps):
				in_stationary = True
				stationary_loss = 0

			if in_stationary:
				stationary_loss += tmp_loss

			# y_assim[i,:] = use_inits

			if any(abs(m_assim_now) > 1000):
				print('Model is blowing up! Abort!')
				print(m_assim_now)
				# n_iters = i
				break

		# pdb.set_trace()
		if in_stationary:
			loss = stationary_loss
		else:
			loss = total_loss

		print('Loss=',loss/i)
		# pw_assim_errors = np.linalg.norm(y_assim - y_clean, axis=1, ord=2)**2
		return (loss.numpy()/i, total_loss.numpy()/i)


	def f_Loss_FullState(G_input, partial_traj, full_traj, n_max, use_inits=inits):
		print('G=',G_input)
		if type(G_input) is not torch.Tensor:
			G_input = torch.FloatTensor(G_input[:,None])

		in_stationary = False
		total_loss = 0
		stationary_loss = np.inf
		y_assim = np.zeros(y_clean.shape)
		for i in range(n_max):
			true_now = torch.FloatTensor(full_traj[i,:,None])
			meas_now = torch.FloatTensor(partial_traj[i,:,None])

			# make prediction using previous state estimate
			sol = solve_ivp(fun=lambda t, y: model(y, t, *model_params['ode_params']), t_span=(tspan[0], tspan[-1]), y0=use_inits.T, method='RK45', t_eval=tspan)
			m_pred_now = torch.FloatTensor(sol.y.T[-1,:,None])
			# y_predictions[i,:] = m_pred_now.detach().numpy().squeeze()

			# Do the assimilation!
			m_assim_now = f_mk(G_input, m_pred_now, meas_now)

			# set inits
			use_inits = m_assim_now.detach().numpy().squeeze()

			# compute loss
			tmp_loss = ( m_assim_now.squeeze() -  true_now.squeeze() ).pow(2).sum()
			total_loss += tmp_loss

			if not in_stationary and (tmp_loss<=eps):
				in_stationary = True
				stationary_loss = 0

			if in_stationary:
				stationary_loss += tmp_loss

			# y_assim[i,:] = use_inits

			if any(abs(m_assim_now) > 1000):
				print('Model is blowing up! Abort!')
				print(m_assim_now)
				# n_iters = i
				break

		# pdb.set_trace()
		if in_stationary:
			loss = stationary_loss
		else:
			loss = total_loss

		print('Loss=',loss/i)
		# pw_assim_errors = np.linalg.norm(y_assim - y_clean, axis=1, ord=2)**2
		return (loss.numpy()/i, total_loss.numpy()/i)


	print('G_assim:', G_assim)
	# print('Total Loss for G:', f_Loss_Sum(G_assim))
	if opt_surfaces_only:
		tvec = [10,50,100]
		tvec_dict = {t: np.int(t/model_params['delta_t']) for t in tvec}
		xgrid = np.linspace(-0.1,0.3,50)
		ygrid = np.linspace(-0.1,0.3,50)
		tmax = max(tvec)
		t_span_opt = np.arange(0, tmax, model_params['delta_t'])
		N_trajs = 1
		for n_traj in range(N_trajs):
			L_grid_PartialState = np.zeros((len(tvec),len(xgrid),len(ygrid)))
			L_grid_FullState = np.zeros((len(tvec),len(xgrid),len(ygrid)))
			L_grid_PartialState_total = np.zeros((len(tvec),len(xgrid),len(ygrid)))
			L_grid_FullState_total = np.zeros((len(tvec),len(xgrid),len(ygrid)))
			G_grid = np.zeros((len(tvec),len(xgrid),len(ygrid),3))
			v0_true = get_lorenz_inits(n=1).squeeze()
			v0_3dvar = get_lorenz_inits(n=1).squeeze()
			sol = solve_ivp(fun=lambda t, y: model(y, t, *model_params['ode_params']), t_span=(t_span_opt[0],t_span_opt[-1]), y0=v0_true, method='RK45', t_eval=t_span_opt)
			true_traj = torch.FloatTensor(sol.y.T)
			true_partial = torch.mm(true_traj,H_obs_lowfi.t())
			noisy_partial_traj = true_partial + eps*torch.FloatTensor(np.random.randn(true_partial.shape[0],true_partial.shape[1]))
			i = -1
			for x in xgrid:
				i += 1
				j = -1
				for y in ygrid:
					j += 1
					print('Solving for (x,y)=',x,y)
					# G = torch.FloatTensor(np.array([[x,y,0.00351079]]).T)
					G = torch.FloatTensor(np.array([[x,y,0.]]).T)
					tc = -1
					for t in tvec_dict:
						tc += 1
						n_max = tvec_dict[t]
						(L_grid_PartialState[tc,i,j], L_grid_PartialState_total[tc,i,j]) = f_Loss_PartialState(G, noisy_partial_traj, n_max=n_max, use_inits=v0_3dvar)
						(L_grid_FullState[tc,i,j], L_grid_FullState_total[tc,i,j]) = f_Loss_FullState(G, noisy_partial_traj, true_traj, n_max=n_max, use_inits=v0_3dvar)
						# L_grid[tc,i,j] = f_Loss_Sum(G)
						G_grid[tc,i,j,:] = G.numpy().squeeze()

			np.save(output_dir+'/L_grid_PartialState_'+str(n_traj), L_grid_PartialState)
			np.save(output_dir+'/L_grid_FullState_'+str(n_traj), L_grid_FullState)
			np.save(output_dir+'/L_grid_PartialStateTotal_'+str(n_traj), L_grid_PartialState_total)
			np.save(output_dir+'/L_grid_FullStateTotal_'+str(n_traj), L_grid_FullState_total)
			np.save(output_dir+'/G_grid', G_grid)

			# stationary loss
			def f_make_plots(outputname, Lpartial, Lfull, Lthresh=1.5, cmap='gist_ncar_r'):
				Lfull = np.copy(Lfull)
				Lpartial = np.copy(Lpartial)
				Lfull[Lfull>Lthresh] = Lthresh
				Lpartial[Lpartial>Lthresh] = Lthresh
				tc = -1
				for my_t in tvec_dict:
					tc += 1
					fig, axlist = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
					ax0 = axlist[0]
					im0 = ax0.imshow(Lfull[tc,:,:].squeeze(), cmap=cmap, interpolation='none', extent=(min(xgrid),max(xgrid),max(ygrid),min(ygrid)))
					ax0.set_xlabel('K_2')
					ax0.set_ylabel('K_1')
					min_val = np.min(Lfull[tc,:,:].squeeze())
					my_min = np.squeeze(G_grid[tc,:].squeeze()[np.where(Lfull[tc,:,:].squeeze()==min_val)])
					ax0.set_title('Full-State \n min={0:.2f} \n @ K1={1:.2f}, K2={2:.2f})'.format(min_val, my_min[0],my_min[1]))
					fig.colorbar(im0, ax=ax0)

					ax1 = axlist[1]
					im1 = ax1.imshow(Lpartial[tc,:,:].squeeze(), cmap=cmap, interpolation='none', extent=(min(xgrid),max(xgrid),max(ygrid),min(ygrid)))
					ax1.set_xlabel('K_2')
					ax1.set_ylabel('K_1')
					min_val = np.min(Lpartial[tc,:,:].squeeze())
					my_min = np.squeeze(G_grid[tc,:].squeeze()[np.where(Lpartial[tc,:,:].squeeze()==min_val)])
					ax1.set_title('Partial-State \n min={0:.2f} \n @ K1={1:.2f}, K2={2:.2f})'.format(min_val, my_min[0],my_min[1]))
					fig.colorbar(im1, ax=ax1)

					fig.suptitle('Trajectory {0}, Length {1}, Stationary-Loss Surface'.format(n_traj, my_t))
					fig.savefig(fname=output_dir+outputname+str(n_traj)+'_length'+str(my_t))
					plt.close(fig)
				return

			f_make_plots('/global_StationaryLoss_surface', L_grid_PartialState, L_grid_FullState, Lthresh=1.5)
			f_make_plots('/global_TotalLoss_surface', L_grid_PartialState_total, L_grid_FullState_total, Lthresh=12)
			f_make_plots('/global_StationaryLoss_surface_noThresh', L_grid_PartialState, L_grid_FullState, Lthresh=np.inf)
			f_make_plots('/global_TotalLoss_surface_noThresh', L_grid_PartialState_total, L_grid_FullState_total, Lthresh=np.inf)


			# now show total loss
			# L_grid_FullState_total[L_grid_FullState_total>12] = 12
			# L_grid_PartialState_total[L_grid_PartialState_total>12] = 12
			# tc = -1
			# cmap = 'gist_ncar_r'
			# for my_t in tvec_dict:
			# 	tc += 1
			# 	fig, axlist = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
			# 	ax0 = axlist[0]
			# 	im0 = ax0.imshow(L_grid_FullState_total[tc,:,:].squeeze(), cmap=cmap, interpolation='none', extent=(min(xgrid),max(xgrid),max(ygrid),min(ygrid)))
			# 	ax0.set_xlabel('K_2')
			# 	ax0.set_ylabel('K_1')
			# 	min_val = np.min(L_grid_FullState_total[tc,:,:].squeeze())
			# 	my_min = np.squeeze(G_grid[tc,:].squeeze()[np.where(L_grid_FullState_total[tc,:,:].squeeze()==min_val)])
			# 	ax0.set_title('Full-State \n min={0:.2f} \n @ K1={1:.2f}, K2={2:.2f})'.format(min_val, my_min[0],my_min[1]))
			# 	fig.colorbar(im0, ax=ax0)

			# 	ax1 = axlist[1]
			# 	im1 = ax1.imshow(L_grid_PartialState_total[tc,:,:].squeeze(), cmap=cmap, interpolation='none', extent=(min(xgrid),max(xgrid),max(ygrid),min(ygrid)))
			# 	ax1.set_xlabel('K_2')
			# 	ax1.set_ylabel('K_1')
			# 	min_val = np.min(L_grid_PartialState_total[tc,:,:].squeeze())
			# 	my_min = np.squeeze(G_grid[tc,:].squeeze()[np.where(L_grid_PartialState_total[tc,:,:].squeeze()==min_val)])
			# 	ax1.set_title('Partial-State \n min={0:.2f} \n @ K1={1:.2f}, K2={2:.2f})'.format(min_val, my_min[0],my_min[1]))
			# 	fig.colorbar(im1, ax=ax1)

			# 	fig.suptitle('Trajectory {0}, Length {1}, Total-Loss Surface'.format(n_traj, my_t))
			# 	fig.savefig(fname=output_dir+'/global_TotalLoss_surface'+str(n_traj)+'_length'+str(my_t))
			# 	plt.close(fig)

			# return


	### optimize G over entire sequence
	G_opt = {}
	if full_sequence:
		# initialize storage variables
		loss_history = np.zeros(n_epochs)
		dL_history = np.zeros(n_epochs)
		G_assim_history = np.zeros((n_epochs, G_assim.shape[0]))
		G_assim_history_running_mean = np.zeros((n_epochs, G_assim.shape[0]))
		if optimization == 'NelderMead':
			if optim_full_state:
				evalG = lambda G: f_Loss_FullState(G, partial_traj=torch.FloatTensor(y_noisy_lowfi), full_traj=torch.FloatTensor(y_clean), n_max=n_iters, use_inits=inits)[0]
			else:
				evalG = lambda G: f_Loss_PartialState(G, partial_traj=torch.FloatTensor(y_noisy_lowfi), n_max=n_iters, use_inits=inits)[0]
			fmin = np.inf
			# G0 = np.array([1,1,1])
			# opt = scipy.optimize.fmin(func=evalG, x0=G0, full_output=True)
			# if opt[1] <= fmin:
			# 	fmin = opt[1]
			# 	Gbest = opt[0]
			# G_opt['opt_struct'] = []
			G_opt['inits'] = np.zeros((n_nelder_inits,G_assim.shape[0]))
			G_opt['final'] = np.zeros((n_nelder_inits,G_assim.shape[0]))
			G_opt['optval'] = np.zeros((n_nelder_inits,G_assim.shape[0]))
			for i_nm in range(n_nelder_inits):
				if random_nelder_inits:
					G0 = np.random.multivariate_normal(mean=[0,0,0], cov=(0.1**2)*np.eye(3)).T[:,None]
				else:
					G0 = G_assim.numpy()
				print('NM Init=',G0)

				# KoptHistory = [] #np.zeros((1, G_assim.shape[0]))
				# def store(G):
				# 	KoptHistory.append(G)
				# opt = scipy.optimize.fmin(func=evalG, x0=G0.squeeze(), callback=store, maxfun=max_nelder_sols, full_output=True, disp=True)

				opt = scipy.optimize.fmin(func=evalG, x0=G0.squeeze(), maxfun=max_nelder_sols, full_output=True, disp=True)
				# G_opt['opt_struct'].append(opt)
				G_opt['inits'][i_nm,:] = G0.squeeze()
				G_opt['final'][i_nm,:] = opt[0].squeeze()
				G_opt['optval'][i_nm,:] = opt[1].squeeze()
				print('NM solution=', opt[0])
				if opt[1] <= fmin:
					fmin = opt[1]
					Gbest = opt[0]
			G_assim_history[:,:] = Gbest
			G_assim_history_running_mean[:,:] = Gbest
		else:
			for kk in range(n_epochs):
				print('Iter',kk,'G_assim Pre:', G_assim)
				use_inits = inits #get_lorenz_inits(n=1).squeeze()
				LkG = f_Loss_Sum(G_assim, use_inits=use_inits)
				# perturb
				Q = np.random.randn(*G_assim.shape)
				Q = torch.FloatTensor(Q/np.linalg.norm(Q,'fro'))
				Gplus = G_assim + h*Q
				Gminus = G_assim - h*Q

				# if any(abs(Gplus)>1):
				# 	print('Gplus is violating constraint. Mapping to constraint boundary.')
				# Gplus[Gplus>1] = 1
				# Gplus[Gplus<-1] = -1

				LkPlus = np.log(f_Loss_Sum(Gplus, use_inits=use_inits))
				LkMinus = np.log(f_Loss_Sum(Gminus, use_inits=use_inits))

				# dL = ( LkPlus - LkG )/h
				dL = ( LkPlus - LkMinus )/(2*h)

				G_assim -= lr_G * dL * Q

				# if any(abs(G_assim)>1):
				# 	print('G_assim is violating constraint. Mapping to constraint boundary.')
				# G_assim[G_assim>1] = 1
				# G_assim[G_assim<-1] = -1

				print('LkMinus:', LkMinus)
				print('LkPlus:', LkPlus)
				print('dL:', dL)
				print('G_assim Post:', G_assim)

				# store progress
				dL_history[kk] = dL
				loss_history[kk] = LkG
				G_assim_history[kk,:] = G_assim.detach().numpy().squeeze()
				G_assim_history_running_mean[kk,:] = np.mean(G_assim_history[:kk,:], axis=0)

			np.savez(output_dir+'/output.npz', G_assim_history=G_assim_history, G_assim_history_running_mean=G_assim_history_running_mean,
				model_params=model_params, eps=eps, loss_history=loss_history, dL_history=dL_history, h=h, lr_G=lr_G)
	else:
		# Standard Gradient Descent
		# initialize storage variables
		loss_history = np.zeros(n_iters)
		dL_history = np.zeros(n_iters)
		G_assim_history = np.zeros((y_clean.shape[0], G_assim.shape[0]))
		G_assim_history_running_mean = np.zeros((y_clean.shape[0], G_assim.shape[0]))
		### NEW code for updating G_assim
		for kk in range(n_epochs):
			use_inits = inits
			for i in range(n_iters):
				meas_now = torch.FloatTensor(y_noisy_lowfi[i,:,None])

				# make prediction using previous state estimate
				# pdb.set_trace()
				sol = solve_ivp(fun=lambda t, y: model(y, t, *model_params['ode_params']), t_span=(tspan[0], tspan[-1]), y0=use_inits.T, method='RK45', t_eval=tspan)
				m_pred_now = torch.FloatTensor(sol.y.T[-1,:,None])
				y_predictions[i,:] = m_pred_now.detach().numpy().squeeze()

				# Do the assimilation!
				try:
					m_assim_now = f_mk(G_assim, m_pred_now, meas_now)
				except:
					pdb.set_trace()
				G_assim_history[i,:] = G_assim.detach().numpy().squeeze()
				G_assim_history_running_mean[i,:] = np.mean(G_assim_history[:i,:], axis=0)
				use_inits = m_assim_now.detach().numpy().squeeze()
				y_assim[i,:] = use_inits

				# compute loss
				if learn_assim:
					if cheat:
						meas_hifi_now = torch.FloatTensor(y_hifi[i,:])
						meas_lowfi_now = meas_now
						loss = f_Lk_cheat(G_assim, m_pred_now, meas_lowfi_now, meas_hifi_now)
						loss.backward()
						G_assim.data -= lr * G_assim.grad.data
						G_assim.grad.data.zero_()
						loss_history[i] = loss
						# print('Iter',i,'Loss:',loss)
					else:
						if new_cheat:
							H = H_obs_hifi
							meas_now = torch.FloatTensor(y_hifi[i,:])
						else:
							H = H_obs_lowfi

						if i>1:
							# Gather historical data for propagation
							m_assim_prev2 = y_assim[i-2,:] #basically \hat{m}_{i-2}
							meas_prev1 = torch.FloatTensor(y_noisy_lowfi[i-1,:,None])

							LkG = f_Lk(G_assim, m_assim_prev2, meas_prev1, meas_now, H=H)
							for iq in range(N_q_tries):
								# sample a G-like matrix for approximating a random directional derivative
								Q = np.random.randn(*G_assim.shape)
								Q = torch.FloatTensor(Q/np.linalg.norm(Q,'fro'))
								Gplus = G_assim + h*Q

								# approximate directional derivative
								LkGplus = f_Lk(Gplus, m_assim_prev2, meas_prev1, meas_now, H=H)
								if iq==0 or (LkGplus < LkGplus_best):
									LkGplus_best = LkGplus
									Q_best = Q


							# update G_assim by random approximate directional derivative
							dL = ( LkGplus_best - LkG )/h
							Gdiff += lr_G * dL * Q_best
							if (i % G_update_interval)==0:
								G_assim.data -= Gdiff
								Gdiff = 0*G_assim.data
							loss_history[i] = LkG
							dL_history[i] = dL
						else:
							Gdiff = 0*G_assim.data
					# save intermittently during training
					if (i % 50) == 0:
						np.savez(output_dir+'/output.npz', G_assim_history=G_assim_history, G_assim_history_running_mean=G_assim_history_running_mean, y_assim=y_assim, y_predictions=y_predictions,
							model_params=model_params, eps=eps, loss_history=loss_history, dL_history=dL_history, h=h, lr_G=lr_G, i=i, kk=kk)


	## Done running 3DVAR, now summarize
	if learn_assim:
		if full_sequence:
			fig, (axlist) = plt.subplots(nrows=3, ncols=1)

			# plot running average of G_assim
			for kk in range(len(plot_state_indices)):
				axlist[0].plot(G_assim_history_running_mean[:,plot_state_indices[kk]],label='G_{0}'.format(plot_state_indices[kk]))
				axlist[1].plot(G_assim_history[:,plot_state_indices[kk]],label='G_{0}'.format(plot_state_indices[kk]))
			axlist[0].legend()
			axlist[1].legend()
			axlist[0].set_title('3DVAR Assimilation Matrix Convergence (Running Mean)')
			axlist[1].set_title('3DVAR Assimilation Matrix Sequence')

			axlist[2].plot(loss_history/n_iters, label='MSE')
			# axlist[2].plot([eps for _ in range(len(loss_history))], color = 'black', linestyle='--', label = r'$\epsilon$')
			axlist[2].set_title('Assimilation MSE')
			axlist[2].set_xlabel('time')
			axlist[2].set_yscale('log')
		else:
			fig, (axlist) = plt.subplots(nrows=2+len(plot_state_indices), ncols=1,
								figsize = [10, 12])

			# plot running average of G_assim
			for kk in range(len(plot_state_indices)):
				t_plot = np.arange(0,round(len(y_clean[:,0])*model_params['delta_t'],8),model_params['delta_t'])
				axlist[0].plot(t_plot, G_assim_history_running_mean[:,plot_state_indices[kk]],label='G_{0}'.format(plot_state_indices[kk]))
				axlist[1].plot(t_plot, G_assim_history[:,plot_state_indices[kk]],label='G_{0}'.format(plot_state_indices[kk]))
			# axlist[2].plot(t_plot, loss_history)
			# axlist[2].plot(t_plot, dL_history)

			axlist[0].legend()
			axlist[1].legend()
			axlist[0].set_xticklabels([])
			# axlist[1].set_xticklabels([])

			axlist[0].set_title('3DVAR Assimilation Matrix Convergence (Running Mean)')
			axlist[1].set_title('3DVAR Assimilation Matrix Sequence')
			# axlist[2].set_title('Loss Sequence')
			# axlist[2].set_ylabel('dL')

			# axlist[2].set_yscale('log')

			for kk in range(len(plot_state_indices)):
				ax = axlist[kk+2]
				t_plot = np.arange(0,round(len(y_clean[:,0])*model_params['delta_t'],8),model_params['delta_t'])
				ax.plot(t_plot, y_clean[:,plot_state_indices[kk]], color='red', label='clean data')
				ax.plot(t_plot, y_predictions[:,plot_state_indices[kk]], color='black', label='3DVAR')
				ax.set_ylabel(model_params['state_names'][plot_state_indices[kk]] + '(t)')

			ax.set_xlabel('time')

		fig.savefig(fname=output_dir+'/3DVAR_assimilation_matrix_learning_convergence')
		plt.close(fig)


	if full_sequence:
		np.savez(output_dir+'/output.npz', G_assim_history=G_assim_history, G_assim_history_running_mean=G_assim_history_running_mean,
		model_params=model_params, eps=eps, loss_history=loss_history, dL_history=dL_history, h=h, lr_G=lr_G, G_opt=G_opt)
		return G_assim_history_running_mean[-1,:]

	## PLOTS
	fig, (ax_list) = plt.subplots(len(plot_state_indices),1)
	if not isinstance(ax_list,np.ndarray):
		ax_list = [ax_list]
	for kk in range(len(ax_list)):
		ax1 = ax_list[kk]
		t_plot = np.arange(0,round(len(y_clean[n_plt_start:,plot_state_indices[kk]])*model_params['delta_t'],8),model_params['delta_t'])
		t_max = round(len(y_clean[:,0])*model_params['delta_t'],8)
		t_plot = t_plot + (t_max - max(t_plot))
		# ax1.scatter(t_plot, y_noisy[:n_plt,plot_state_indices[kk]], color='red', s=10, alpha=0.3, label='noisy data')
		ax1.plot(t_plot, y_clean[n_plt_start:,plot_state_indices[kk]], color='red', label='clean data')
		ax1.plot(t_plot, y_predictions[n_plt_start:,plot_state_indices[kk]], color='black', label='3DVAR')
		ax1.set_xlabel('time')
		ax1.set_ylabel(model_params['state_names'][plot_state_indices[kk]] + '(t)')
	ax_list[0].legend()
	fig.suptitle('3DVAR output')
	fig.savefig(fname=output_dir+'/3DVAR_timeseries')
	plt.close(fig)

	fig, (ax0, ax1) = plt.subplots(nrows=2,ncols=1, sharex=True)

	# ax0 is for assimilation errors...Cumulative AVG and Pointwise, meas-only and all states
	# ax1 is for prediction errors...Cumulative AVG and Pointwise, all states
	t_plot = np.arange(0,round(len(y_clean[:,0])*model_params['delta_t'],8),model_params['delta_t'])

	# these give the mean squared error
	pw_assim_errors = np.linalg.norm(y_assim - y_clean, axis=1, ord=2)**2
	pw_assim_errors_OBS = np.linalg.norm(np.matmul(H_obs_lowfi.numpy(),y_assim.T).T - y_clean_lowfi, axis=1, ord=2)**2
	pw_pred_errors = np.linalg.norm(y_predictions - y_clean, axis=1, ord=2)**2
	pw_pred_errors_OBS = np.linalg.norm(np.matmul(H_obs_lowfi.numpy(),y_predictions.T).T - y_clean_lowfi, axis=1, ord=2)**2

	print('Mean(pw_assim_errors) = ', np.mean(pw_assim_errors))

	running_pred_errors = np.zeros(pw_pred_errors.shape)
	running_pred_errors_OBS = np.zeros(pw_pred_errors.shape)
	running_assim_errors = np.zeros(pw_pred_errors.shape)
	running_assim_errors_OBS = np.zeros(pw_pred_errors.shape)

	for k in range(running_pred_errors.shape[0]):
		running_pred_errors[k] = np.mean(pw_pred_errors[:(k+1)])
		running_pred_errors_OBS[k] = np.mean(pw_pred_errors_OBS[:(k+1)])
		running_assim_errors[k] = np.mean(pw_assim_errors[:(k+1)])
		running_assim_errors_OBS[k] = np.mean(pw_assim_errors_OBS[:(k+1)])

	ax0.plot(t_plot, pw_pred_errors, color='blue', label='Full-State Point-wise')
	if eps:
		ax0.plot(t_plot, [eps for _ in range(len(t_plot))], color = 'black', linestyle='--', label = r'$\epsilon$')
	# ax0.scatter(t_plot, pw_pred_errors_OBS[:n_plt], color='blue', label='Observed-State Point-wise')
	# ax0.plot(t_plot, running_pred_errors, linestyle='--', color='black', label='Full-State Cumulative')
	# ax0.plot(t_plot, running_pred_errors_OBS[:n_plt], linestyle=':', color='black', label='Observed-State Cumulative')
	ax0.set_ylabel('MSE')
	ax0.legend()
	ax0.set_title('1-step Prediction Errors')

	ax1.plot(t_plot, pw_assim_errors, linestyle='--', color='blue', label='Full-State Point-wise')
	if eps:
		ax1.plot(t_plot, [eps for _ in range(len(t_plot))], color = 'black', linestyle='--', label = r'$\epsilon$')
	# ax1.plot(t_plot, running_assim_errors, linestyle='--', color='black', label='Full-State Cumulative')
	ax1.set_ylabel('MSE')
	ax1.legend()
	ax1.set_title('Assimilation Errors')
	ax1.set_xlabel('time')

	fig.suptitle('3DVAR error convergence')
	fig.savefig(fname=output_dir+'/3DVAR_error_convergence')

	ax0.set_yscale('log')
	ax0.set_ylabel('log MSE')
	ax1.set_yscale('log')
	ax1.set_ylabel('log MSE')
	fig.savefig(fname=output_dir+'/3DVAR_error_convergence_log')
	plt.close(fig)

	np.savez(output_dir+'/output.npz', G_assim_history=G_assim_history, G_assim_history_running_mean=G_assim_history_running_mean, y_assim=y_assim, y_predictions=y_predictions,
		pw_assim_errors=pw_assim_errors, pw_assim_errors_OBS=pw_assim_errors_OBS, pw_pred_errors=pw_pred_errors, pw_pred_errors_OBS=pw_pred_errors_OBS,
		model_params=model_params, eps=eps, loss_history=loss_history, dL_history=dL_history, h=h, lr_G=lr_G)

	return G_assim
