## This script does the following
#1. Generates a random input sequence x
#2. Simulates data using a driven exponential decay ODE model
#3. Trains a single-layer RNN using clean data output from ODE and the input sequence
#4. RESULT: Gradients quickly go to 0 when training the RNN

# based off of code from https://www.cpuheater.com/deep-learning/introduction-to-recurrent-neural-networks-in-pytorch/
import os
from time import time
import math
import numpy as np
import numpy.matlib
from scipy.stats import entropy
from scipy.integrate import odeint
# from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.stats import gaussian_kde

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
	y_clean = odeint(model, y0, tspan, args=my_args)
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
	y_clean, y_noisy, x  = run_ode_model(model, tspan, sim_model_params, noise_frac=noise_frac, output_dir=output_dir, drive_system=drive_system, plot_state_indices=plot_state_indices)
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

### RNN fitting section
def forward_vanilla(data_input, hidden_state, w1, w2, b, c, v, *args, **kwargs):
	hidden_state = torch.relu(b + torch.mm(w2,data_input) + torch.mm(w1,hidden_state))
	out = c + torch.mm(v,hidden_state)
	return  (out, hidden_state)

def forward_chaos_pureML(data_input, hidden_state, A, B, C, a, b, *args, **kwargs):
	hidden_state = torch.relu(a + torch.mm(A,hidden_state) + torch.mm(B,data_input))
	# hidden_state = torch.relu(a + torch.mm(A,hidden_state))
	out = b + torch.mm(C,hidden_state)
	return  (out, hidden_state)

def forward_chaos_pureML2(data_input, hidden_state, A, B, C, a, b, *args, **kwargs):
	hidden_state = torch.tanh(a + torch.mm(A,hidden_state))
	out = b + torch.mm(C,hidden_state)
	return  (out, hidden_state)


def forward_chaos_hybrid_full(model_input, hidden_state, A, B, C, a, b, normz_info, model, model_params, model_output=None):
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
		tspan = [0, 0.5*model_params['delta_t'], model_params['delta_t']]
		# driver = xmean + xsd*model_input.detach().numpy()
		# my_args = model_params + (driver,)
		#
		# unnormalize model_input so that it can go through the ODE solver
		y0 = f_unNormalize_minmax(normz_info, y0_normalized.numpy())
		y_out = odeint(model, y0, tspan, args=model_params['ode_params'])

		y_pred = y_out[-1,:] #last column
		y_pred_normalized = f_normalize_minmax(normz_info, y_pred)
	else:
		y_pred_normalized = model_output

	# renormalize
	# hidden_state[0] = torch.from_numpy( (y_out[-1] - ymin) / (ymax - ymin) )
	# hidden_state[0] = torch.from_numpy( (y_out[-1] - ymean) / ysd )

	stacked_input = torch.FloatTensor(np.hstack( (y_pred_normalized, y0_normalized) )[:,None])
	hidden_state = torch.relu( a + torch.mm(A,hidden_state) + torch.mm(B,stacked_input) )
	stacked_output = torch.cat( ( torch.FloatTensor(y_pred_normalized[:,None]), hidden_state ) )
	out = b + torch.mm(C,stacked_output)
	return  (out, hidden_state)


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
	y_out = odeint(model, y0, tspan, args=my_args)

	# renormalize
	hidden_state[0] = torch.from_numpy( (y_out[-1] - ymin) / (ymax - ymin) )
	# hidden_state[0] = torch.from_numpy( (y_out[-1] - ymean) / ysd )

	hidden_state = torch.relu(b + torch.mm(w2,input) + torch.mm(w1,hidden_state))
	out = c + torch.mm(v,hidden_state)
	return  (out, hidden_state)


def run_GP(y_clean_train, y_noisy_train,
			y_clean_test, y_noisy_test,
			model,f_unNormalize_Y,
			model_pred,
			train_seq_length,
			test_seq_length,
			output_size,
			avg_output_test,
			avg_output_clean_test,
			normz_info, model_params,
			plot_state_indices,
			output_dir,
			n_plttrain,
			n_plttest,
			err_thresh, input_style=1):

	if input_style==1:
		X = y_noisy_train[:-1]
	elif input_style==2:
		X = np.concatenate((y_noisy_train[:-1],model_pred),axis=1)

	# NEW. learn residuals with GP
	gpr = GaussianProcessRegressor().fit(X=X,y=y_noisy_train[1:]-model_pred)
	# gpr = GaussianProcessRegressor().fit(X=output_train[:-1],y=output_train[1:]-model_pred)
	gpr_train_predictions_orig = model_pred + gpr.predict(X).squeeze()
	# gpr.score(model_pred, y_noisy_train[1:])
	total_loss_test = 0
	total_loss_clean_test = 0
	pred = np.squeeze(y_noisy_train[-1,:,None].T)
	pw_loss_test = np.zeros(test_seq_length)
	pw_loss_clean_test = np.zeros(test_seq_length)
	gpr_test_predictions = np.zeros([test_seq_length, output_size])
	for j in range(test_seq_length):
		target = y_noisy_test[j,None].squeeze()
		target_clean = y_clean_test[j,None].squeeze()

		# generate next-step ODE model prediction
		tspan = [0, 0.5*model_params['delta_t'], model_params['delta_t']]
		# unnormalize model_input so that it can go through the ODE solver
		y0 = f_unNormalize_minmax(normz_info, pred)
		y_out = odeint(model, y0, tspan, args=model_params['ode_params'])
		my_model_pred = f_normalize_minmax(normz_info, y_out[-1,:])

		if input_style==1:
			x = pred
		elif input_style==2:
			x = np.concatenate((pred, my_model_pred))
		pred = my_model_pred + gpr.predict(x.reshape(1, -1) , return_std=False).squeeze()

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

	gpr_train_predictions_rerun = np.zeros([train_seq_length-1, output_size])
	pred = y_noisy_train[0,:,None].squeeze()
	for j in range(train_seq_length-1):
		# target = output_test[j,None]
		# target_clean = output_clean_train[j,None]

		# generate next-step ODE model prediction
		tspan = [0, 0.5*model_params['delta_t'], model_params['delta_t']]
		# unnormalize model_input so that it can go through the ODE solver
		y0 = f_unNormalize_minmax(normz_info, pred)
		y_out = odeint(model, y0, tspan, args=model_params['ode_params'])
		my_model_pred = f_normalize_minmax(normz_info, y_out[-1,:])

		if input_style==1:
			x = pred
		elif input_style==2:
			x = np.concatenate((pred, my_model_pred))

		pred = my_model_pred + gpr.predict(x.reshape(1, -1), return_std=False).squeeze()

		# compute losses
		# total_loss_test += (pred - target.squeeze()).pow(2).sum()/2
		# total_loss_clean_test += (pred - target_clean.squeeze()).pow(2).sum()/2
		# pw_loss_test[j] = total_loss_test / avg_output_test
		# pw_loss_clean_test[j] = total_loss_clean_test / avg_output_clean_test
		gpr_train_predictions_rerun[j,:] = pred

	# plot forecast over test set
	y_clean_test_raw = f_unNormalize_Y(normz_info,y_clean_test)
	y_noisy_train_raw = f_unNormalize_Y(normz_info,y_noisy_train)
	y_clean_train_raw = f_unNormalize_Y(normz_info,y_clean_train)
	gpr_train_predictions_orig_raw = f_unNormalize_Y(normz_info,gpr_train_predictions_orig)
	gpr_train_predictions_rerun_raw = f_unNormalize_Y(normz_info,gpr_train_predictions_rerun)
	gpr_test_predictions_raw = f_unNormalize_Y(normz_info,gpr_test_predictions)

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

	fig.suptitle('GPR {0} Training Fit'.format(input_style))
	fig.savefig(fname=output_dir+'/GPR_{0}_train_fit_ode'.format(input_style))
	plt.close(fig)


	fig, (ax_list) = plt.subplots(len(plot_state_indices),1)
	if not isinstance(ax_list,np.ndarray):
		ax_list = [ax_list]

	# NOW, show testing fit
	for kk in range(len(ax_list)):
		ax3 = ax_list[kk]
		t_plot = np.arange(0,len(y_clean_test[:n_plttest,plot_state_indices[kk]])*model_params['delta_t'],model_params['delta_t'])
		# ax3.scatter(np.arange(len(y_noisy_test)), y_noisy_test, color='red', s=10, alpha=0.3, label='noisy data')
		ax3.plot(t_plot, y_clean_test_raw[:n_plttest,plot_state_indices[kk]], color='red', label='clean data')
		ax3.plot(t_plot, gpr_test_predictions_raw[:n_plttest,plot_state_indices[kk]], color='black', label='NN fit')
		ax3.set_xlabel('time')
		ax3.set_ylabel(model_params['state_names'][plot_state_indices[kk]] +'(t)', color='red')
		ax3.tick_params(axis='y', labelcolor='red')

	ax_list[0].legend()

	fig.suptitle('GPR {0} Testing fit'.format(input_style))
	fig.savefig(fname=output_dir+'/GPR_{0}_test_fit_ode'.format(input_style))
	plt.close(fig)

	# write pw losses to file
	gpr_pred_validity_vec_test = np.argmax(pw_loss_test > err_thresh)*model_params['delta_t']
	gpr_pred_validity_vec_clean_test = np.argmax(pw_loss_clean_test > err_thresh)*model_params['delta_t']
	with open(output_dir+'/GPR_{0}_prediction_validity_time_test.txt'.format(input_style), "w") as f:
		f.write(str(gpr_pred_validity_vec_test))
	with open(output_dir+'/GPR_{0}_prediction_validity_time_clean_test.txt'.format(input_style), "w") as f:
		f.write(str(gpr_pred_validity_vec_clean_test))
	with open(output_dir+'/GPR_{0}_loss_test.txt'.format(input_style), "w") as f:
		f.write(str(total_loss_test))
	with open(output_dir+'/GPR_{0}_clean_loss_test.txt'.format(input_style), "w") as f:
		f.write(str(total_loss_clean_test))



def train_chaosRNN(forward,
			y_clean_train, y_noisy_train,
			y_clean_test, y_noisy_test,
			model_params, hidden_size=6, n_epochs=100, lr=0.05,
			output_dir='.', normz_info=None, model=None,
			trivial_init=False, perturb_trivial_init=True, sd_perturb = 0.001,
			stack_hidden=True, stack_output=True,
			x_train=None, x_test=None,
			f_normalize_Y=f_normalize_minmax,
			f_unNormalize_Y=f_unNormalize_minmax,
			f_normalize_X = f_normalize_ztrans,
			f_unNormalize_X = f_unNormalize_ztrans,
			max_plot=2000, n_param_saves=None,
			err_thresh=0.4, plot_state_indices=None,
			precompute_model=True, kde_func=kde_scipy,
			compute_kl=False):

	if torch.cuda.is_available():
		print('Using CUDA FloatTensor')
		dtype = torch.cuda.FloatTensor
	else:
		print('Using regular torch.FloatTensor')
		dtype = torch.FloatTensor

	n_plttrain = min(max_plot,y_clean_train.shape[0])
	n_plttest = min(max_plot,y_clean_test.shape[0])

	if not plot_state_indices:
		plot_state_indices = np.arange(y_clean_test.shape[1])

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

	avg_output_test = torch.mean(output_test**2).detach().numpy()**0.5
	# avg_output_test = torch.mean(output_test**2,dim=(0,1)).detach().numpy()**0.5
	avg_output_clean_test = torch.mean(output_clean_test**2).detach().numpy()**0.5
	# avg_output_clean_test = torch.mean(output_clean_test**2,dim=(0,1)).detach().numpy()**0.5

	output_size = output_train.shape[1]
	train_seq_length = output_train.size(0)
	test_seq_length = output_test.size(0)

	# compute one-step-ahead model-based prediction for each point in the training set
	if precompute_model:
		model_pred = np.zeros((train_seq_length-1,output_size))
		for j in range(train_seq_length-1):
			tspan = [0, 0.5*model_params['delta_t'], model_params['delta_t']]
			# unnormalize model_input so that it can go through the ODE solver
			y0 = f_unNormalize_minmax(normz_info, output_train[j,:].numpy())
			y_out = odeint(model, y0, tspan, args=model_params['ode_params'])
			model_pred[j,:] = f_normalize_minmax(normz_info, y_out[-1,:])
	else:
		model_pred = [None for j in range(train_seq_length-1)]

	for input_style in [2,1]:
		run_GP(y_clean_train, y_noisy_train,
				y_clean_test, y_noisy_test,
				model,f_unNormalize_Y,
				model_pred,
				train_seq_length,
				test_seq_length,
				output_size,
				avg_output_test,
				avg_output_clean_test,
				normz_info, model_params,
				plot_state_indices,
				output_dir,
				n_plttrain,
				n_plttest,
				err_thresh,
				input_style)

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


		hidden_state = torch.zeros((hidden_size, 1)).type(dtype)
		predictions = np.zeros([test_seq_length, output_size])
		# yb_normalized = (yb - YMIN)/(YMAX - YMIN)
		# initializing y0 of hidden state to the true initial condition from the clean signal
		# hidden_state[0] = float(y_clean_test[0])
		pred = output_train[-1,:,None]
		perf_total_loss_test = 0
		# running_epoch_loss_test = np.zeros(test_seq_length)
		perf_pw_loss_test = np.zeros(test_seq_length)
		for i in range(test_seq_length):
			(pred, hidden_state) = forward(pred, hidden_state, A,B,C,a,b , normz_info, model, model_params)
			# hidden_state = hidden_state
			predictions[i,:] = pred.data.numpy().ravel()
			i_loss = (pred.detach().squeeze() - output_test[i,None].squeeze()).pow(2).sum()
			perf_total_loss_test += i_loss
			# running_epoch_loss_test[j] = total_loss_test/(j+1)
			perf_pw_loss_test[i] = i_loss.pow(0.5).numpy() / avg_output_test

		# get error metrics
		perfect_test_loss = perf_total_loss_test.numpy() / test_seq_length
		perfect_t_valid = np.argmax(perf_pw_loss_test > err_thresh)*model_params['delta_t']

		with open(output_dir+'/perfectModel_loss_test.txt', "w") as f:
			f.write(str(perfect_test_loss))
		with open(output_dir+'/perfectModel_validity_time_test.txt', "w") as f:
			f.write(str(perfect_t_valid))

		# plot predictions vs truth
		fig, (ax_list) = plt.subplots(len(plot_state_indices),1)
		if not isinstance(ax_list,np.ndarray):
			ax_list = [ax_list]

		for kk in range(len(ax_list)):
			ax1 = ax_list[kk]
			ax1.scatter(np.arange(len(y_noisy_test[:n_plttest,plot_state_indices[kk]])), y_noisy_test[:n_plttest,plot_state_indices[kk]], color='red', s=10, alpha=0.3, label='noisy data')
			ax1.plot(y_clean_test[:n_plttest,kk], color='red', label='clean data')
			ax1.plot(predictions[:n_plttest,kk], ':' ,color='red', label='NN trivial fit')
			ax1.set_xlabel('time')
			ax1.set_ylabel(model_params['state_names'][kk] + '(t)', color='red')
			ax1.tick_params(axis='y', labelcolor='red')

		ax_list[0].legend()
		fig.suptitle('RNN w/ just mechanism fit to ODE simulation TEST SET')
		fig.savefig(fname=output_dir+'/PERFECT_MechRnn_fit_ode')
		plt.close(fig)


	# Initilize parameters for training
	A = torch.zeros(hidden_size, hidden_size).type(dtype)
	B = torch.zeros(hidden_size, (1+stack_hidden)*output_size).type(dtype)
	a = torch.zeros(hidden_size, 1).type(dtype)
	C = torch.zeros(output_size, hidden_size + (stack_output*output_size) ).type(dtype)
	b = torch.zeros(output_size, 1).type(dtype)

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
	loss_vec_test = np.zeros(n_epochs)
	loss_vec_clean_test = np.zeros(n_epochs)
	pred_validity_vec_test = np.zeros(n_epochs)
	pred_validity_vec_clean_test = np.zeros(n_epochs)
	kl_vec_inv_test = np.zeros((n_epochs, output_size))
	kl_vec_inv_clean_test = np.zeros((n_epochs, output_size))
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
		for j in range(train_seq_length-1):
			cc += 1
			target = output_train[j+1,None]
			target_clean = output_clean_train[j+1,None]
			(pred, hidden_state) = forward(output_train[j,:,None], hidden_state, A,B,C,a,b, normz_info, model, model_params, model_output=model_pred[j])
			# (pred, hidden_state) = forward(pred.detach(), hidden_state, A,B,C,a,b, normz_info, model, model_params)
			loss = (pred.squeeze() - target.squeeze()).pow(2).sum()/2
			total_loss_train += loss
			total_loss_clean_train += (pred.squeeze() - target_clean.squeeze()).pow(2).sum()/2
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

		total_loss_test = 0
		total_loss_clean_test = 0
		running_epoch_loss_test = np.zeros(test_seq_length)
		running_epoch_loss_clean_test = np.zeros(test_seq_length)
		# hidden_state = Variable(torch.zeros((hidden_size, 1)).type(dtype), requires_grad=False)
		pred = output_train[-1,:,None]
		pw_loss_test = np.zeros(test_seq_length)
		pw_loss_clean_test = np.zeros(test_seq_length)
		long_predictions = np.zeros([test_seq_length, output_size])
		for j in range(test_seq_length):
			target = output_test[j,None]
			target_clean = output_clean_test[j,None]
			(pred, hidden_state) = forward(pred, hidden_state, A,B,C,a,b, normz_info, model, model_params)
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
		loss_vec_test[i_epoch] = total_loss_test
		loss_vec_clean_test[i_epoch] = total_loss_clean_test
		pred_validity_vec_test[i_epoch] = np.argmax(pw_loss_test > err_thresh)*model_params['delta_t']
		pred_validity_vec_clean_test[i_epoch] = np.argmax(pw_loss_clean_test > err_thresh)*model_params['delta_t']

		# compute KL divergence between long predictions and whole test set:
		if compute_kl:
			kl_vec_inv_test[i_epoch,:] = kl4dummies(
							f_unNormalize_Y(normz_info, y_noisy_test),
							f_unNormalize_Y(normz_info, long_predictions))
			kl_vec_inv_clean_test[i_epoch,:] = kl4dummies(
							f_unNormalize_Y(normz_info, y_clean_test),
							f_unNormalize_Y(normz_info, long_predictions))

		# print updates every 10 iterations or in 10% incrememnts
		if i_epoch % int( max(2, np.ceil(n_epochs/10)) ) == 0:
			print("Epoch: {}\nTraining Loss = {}\nTesting Loss = {}".format(
						i_epoch,
						total_loss_train.data.item(),
						total_loss_test.data.item()))
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
			for i in range(train_seq_length-1):
				(pred, hidden_state) = forward(output_train[i,:,None], hidden_state, A,B,C,a,b, normz_info, model, model_params, model_output=model_pred[i])
				# (pred, hidden_state) = forward(pred, hidden_state, A,B,C,a,b, normz_info, model, model_params)
				# hidden_state = hidden_state
				saved_hidden_states[i+1,:] = hidden_state.data.numpy().ravel()
				predictions[i+1,:] = pred.data.numpy().ravel()

			y_clean_test_raw = f_unNormalize_Y(normz_info,y_clean_test)
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


			fig, (ax_list) = plt.subplots(len(plot_state_indices),1)
			if not isinstance(ax_list,np.ndarray):
				ax_list = [ax_list]

			# NOW, show testing fit
			# hidden_state = Variable(torch.zeros((hidden_size, 1)).type(dtype), requires_grad=False)
			predictions = np.zeros([test_seq_length, output_size])
			pred = output_train[-1,:,None]
			saved_hidden_states = np.zeros([test_seq_length, hidden_size])
			for i in range(test_seq_length):
				(pred, hidden_state) = forward(pred, hidden_state, A,B,C,a,b, normz_info, model, model_params)
				# hidden_state = hidden_state
				predictions[i,:] = pred.data.numpy().ravel()
				saved_hidden_states[i,:] = hidden_state.data.numpy().ravel()

			predictions_raw = f_unNormalize_Y(normz_info,predictions)
			for kk in range(len(ax_list)):
				ax3 = ax_list[kk]
				t_plot = np.arange(0,len(y_clean_test[:n_plttest,plot_state_indices[kk]])*model_params['delta_t'],model_params['delta_t'])
				# ax3.scatter(np.arange(len(y_noisy_test)), y_noisy_test, color='red', s=10, alpha=0.3, label='noisy data')
				ax3.plot(t_plot, y_clean_test_raw[:n_plttest,plot_state_indices[kk]], color='red', label='clean data')
				ax3.plot(t_plot, predictions_raw[:n_plttest,plot_state_indices[kk]], color='black', label='NN fit')
				ax3.set_xlabel('time')
				ax3.set_ylabel(model_params['state_names'][plot_state_indices[kk]] +'(t)', color='red')
				ax3.tick_params(axis='y', labelcolor='red')

			ax_list[0].legend()

			fig.suptitle('RNN TEST fit to ODE simulation--' + str(i_epoch) + 'training epochs')
			fig.savefig(fname=output_dir+'/rnn_test_fit_ode_iterEpochs'+str(i_epoch))
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
			fig.savefig(fname=output_dir+'/rnn_test_hidden_states_iterEpochs'+str(i_epoch))
			plt.close(fig)

			# plot KDE of test data vs predictiosn
			fig, (ax_list) = plt.subplots(len(plot_state_indices),1)
			if not isinstance(ax_list,np.ndarray):
				ax_list = [ax_list]

			for kk in range(len(ax_list)):
				ax1 = ax_list[kk]
				pk = plot_state_indices[kk]
				x_grid = np.linspace(min(y_clean_test_raw[:,pk]), max(y_clean_test_raw[:,pk]), 1000)
				ax1.plot(x_grid, kde_func(y_clean_test_raw[:,pk], x_grid), label='clean data')
				x_grid = np.linspace(min(predictions_raw[:,pk]), max(predictions_raw[:,pk]), 1000)
				ax1.plot(x_grid, kde_func(predictions_raw[:,pk], x_grid), label='RNN fit')
				ax1.set_xlabel(model_params['state_names'][pk])

			ax_list[0].legend()

			fig.suptitle('Predictions of Invariant Density')
			fig.savefig(fname=output_dir+'/rnn_test_invDensity_iterEpochs'+str(i_epoch))
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
		np.savetxt(output_dir+'/kl_vec_inv_test.txt',kl_vec_inv_test)
		np.savetxt(output_dir+'/kl_vec_inv_clean_test.txt',kl_vec_inv_clean_test)

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
	for i in range(train_seq_length-1):
		(pred, hidden_state) = forward(output_train[i,:,None], hidden_state, A,B,C,a,b, normz_info, model, model_params, model_output=model_pred[i])
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


	# NOW, show testing fit
	# hidden_state = Variable(torch.zeros((hidden_size, 1)).type(dtype), requires_grad=False)
	predictions = np.zeros([test_seq_length, output_size])
	pred = output_train[-1,:,None]
	for i in range(test_seq_length):
		(pred, hidden_state) = forward(pred, hidden_state, A,B,C,a,b, normz_info, model, model_params)
		# hidden_state = hidden_state
		predictions[i,:] = pred.data.numpy().ravel()

	fig, (ax_list) = plt.subplots(len(plot_state_indices),1)
	if not isinstance(ax_list,np.ndarray):
		ax_list = [ax_list]
	predictions_raw = f_unNormalize_Y(normz_info,predictions)
	for kk in range(len(ax_list)):
		ax3 = ax_list[kk]
		t_plot = np.arange(0,len(y_clean_test_raw[:n_plttest,plot_state_indices[kk]])*model_params['delta_t'],model_params['delta_t'])
		# ax3.scatter(np.arange(len(y_noisy_test)), y_noisy_test, color='red', s=10, alpha=0.3, label='noisy data')
		ax3.plot(t_plot, y_clean_test_raw[:n_plttest,plot_state_indices[kk]], color='red', label='clean data')
		ax3.plot(t_plot, predictions_raw[:n_plttest,plot_state_indices[kk]], color='black', label='NN fit')
		ax3.set_xlabel('time')
		ax3.set_ylabel(model_params['state_names'][plot_state_indices[kk]] +'(t)', color='red')
		ax3.tick_params(axis='y', labelcolor='red')
		# ax3.set_title('Testing Fit')

	ax_list[0].legend()
	fig.suptitle('RNN TEST fit to ODE simulation')
	fig.savefig(fname=output_dir+'/rnn_fit_ode_TEST')
	plt.close(fig)

	# plot KDE of test data vs predictiosn
	fig, (ax_list) = plt.subplots(len(plot_state_indices),1)
	if not isinstance(ax_list,np.ndarray):
		ax_list = [ax_list]

	for kk in range(len(ax_list)):
		ax1 = ax_list[kk]
		pk = plot_state_indices[kk]
		x_grid = np.linspace(min(y_clean_test_raw[:,pk]), max(y_clean_test_raw[:,pk]), 1000)
		ax1.plot(x_grid, kde_func(y_clean_test_raw[:,pk], x_grid), label='clean data')
		x_grid = np.linspace(min(predictions_raw[:,pk]), max(predictions_raw[:,pk]), 1000)
		ax1.plot(x_grid, kde_func(predictions_raw[:,pk], x_grid), label='RNN fit')
		ax1.set_xlabel(model_params['state_names'][pk])

	ax_list[0].legend()

	fig.suptitle('Predictions of Invariant Density')
	fig.savefig(fname=output_dir+'/rnn_invDensity_TEST')
	plt.close(fig)


	# plot Train/Test curve
	x_test = pd.DataFrame(np.loadtxt(output_dir+"/loss_vec_clean_test.txt"))
	n_vals = len(x_test)
	max_exp = int(np.floor(np.log10(n_vals)))
	win_list = [None] + list(10**np.arange(1,max_exp))
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
		gpr1_valid_test = np.loadtxt(output_dir+'/GPR_1_prediction_validity_time_clean_test.txt')
		gpr2_valid_test = np.loadtxt(output_dir+'/GPR_2_prediction_validity_time_clean_test.txt')
		gpr1_test = np.loadtxt(output_dir+'/GPR_1_clean_loss_test.txt')
		gpr2_test = np.loadtxt(output_dir+'/GPR_2_clean_loss_test.txt')

		if n_epochs > 1:
			x_valid_test = pd.DataFrame(np.loadtxt(output_dir+"/prediction_validity_time_clean_test.txt"))
			x_kl_test = pd.DataFrame(np.loadtxt(output_dir+"/kl_vec_inv_clean_test.txt"))
		if win:
			ax1.plot(x_train.rolling(win).mean())
			ax2.plot(x_test.rolling(win).mean(),label='RNN')
			if n_epochs > 1:
				ax3.plot(x_valid_test.rolling(win).mean(),label='RNN')
				for kk in plot_state_indices:
					ax4.plot(x_kl_test.loc[:,kk].rolling(win).mean(), label=model_params['state_names'][kk])
		else:
			ax1.plot(x_train)
			ax2.plot(x_test,label='RNN')
			if n_epochs > 1:
				ax3.plot(x_valid_test)
				for kk in plot_state_indices:
					ax4.plot(x_kl_test.loc[:,kk], label=model_params['state_names'][kk])

		ax2.hlines(gpr1_test,xmin=0,xmax=n_vals-1,label='GPR 1',color='red')
		ax2.hlines(gpr2_test,xmin=0,xmax=n_vals-1,label='GPR 2',color='purple')
		ax3.hlines(gpr1_valid_test,xmin=0,xmax=n_vals-1,label='GPR 1',color='red')
		ax3.hlines(gpr2_valid_test,xmin=0,xmax=n_vals-1,label='GPR 2',color='purple')

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
			loss = (pred.squeeze() - target.squeeze()).pow(2).sum()/2
			total_loss_train += loss
			total_loss_clean_train += (pred.squeeze() - target_clean.squeeze()).pow(2).sum()/2
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
			total_loss_test += (pred.squeeze() - target.squeeze()).pow(2).sum()/2
			total_loss_clean_test += (pred.squeeze() - target_clean.squeeze()).pow(2).sum()/2

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
	n_vals = len(x_test)
	max_exp = int(np.floor(np.log10(n_vals)))
	win_list = [None] + list(10**np.arange(1,max_exp))

	try:
		many_epochs = True
		x_kl_test = pd.DataFrame(np.loadtxt(my_dirs[0]+"/kl_vec_inv_clean_test.txt"))
		if not plot_state_indices:
			plot_state_indices = np.arange(x_kl_test.shape[1])
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
			x_train = pd.DataFrame(np.loadtxt(d+"/loss_vec_train.txt"))
			x_test = pd.DataFrame(np.loadtxt(d+"/loss_vec_clean_test.txt"))

			# get GPR fits
			gpr1_valid_test = np.loadtxt(d+'/GPR_1_prediction_validity_time_clean_test.txt')
			gpr2_valid_test = np.loadtxt(d+'/GPR_2_prediction_validity_time_clean_test.txt')
			gpr1_test = np.loadtxt(d+'/GPR_1_clean_loss_test.txt')
			gpr2_test = np.loadtxt(d+'/GPR_2_clean_loss_test.txt')

			if many_epochs:
				x_valid_test = pd.DataFrame(np.loadtxt(d+"/prediction_validity_time_clean_test.txt"))
				x_kl_test = pd.DataFrame(np.loadtxt(d+"/kl_vec_inv_clean_test.txt"))
			if win:
				ax1.plot(x_train.rolling(win).mean(), label=d_label)
				ax2.plot(x_test.rolling(win).mean(), label=d_label)
				if many_epochs:
					ax3.plot(x_valid_test.rolling(win).mean(), label=d_label)
					for kk in plot_state_indices:
						ax4.plot(x_kl_test.loc[:,kk].rolling(win).mean(), label=d_label)
			else:
				ax1.plot(x_train, label=d_label)
				ax2.plot(x_test, label=d_label)
				if many_epochs:
					ax3.plot(x_valid_test, label=d_label)
					for kk in plot_state_indices:
						ax4.plot(x_kl_test.loc[:,kk], label=d_label)

			try:
				gp_label = [x for x in d_label.split('_') if 'epsBad' in x][0]
				ax2.plot([0,n_vals-1],[gpr1_test,gpr1_test],label='GPR 1 ' + gp_label)
				ax2.plot([0,n_vals-1],[gpr2_test,gpr2_test],label='GPR 2 '+gp_label)
				ax3.plot([0,n_vals-1],[gpr1_valid_test,gpr1_valid_test],label='GPR 1 ' + gp_label)
				ax3.plot([0,n_vals-1],[gpr2_valid_test,gpr2_valid_test],label='GPR 2 ' + gp_label)
			except:
				pass

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



def extract_epsilon_performance(my_dirs, output_fname="./epsilon_comparisons", win=10, many_epochs=True, eps_token='epsBadness'):
	t_vec = [1,2,4,6,8,10]
	# first, get sizes of things...max window size is 10% of whole test set.
	d_label = my_dirs[0].split("/")[-1].rstrip('_noisy').rstrip('_clean')
	x_test = pd.DataFrame(np.loadtxt(my_dirs[0]+"/loss_vec_clean_test.txt"))
	n_vals = len(x_test)

	win = min(win,n_vals//3)
	model_performance = {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}}
	rnn_performance = {'mse':(), 't_valid':(), 'mse_time':(), 't_valid_time':{}}
	hybrid_performance = {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}}
	gpr1_performance = {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}}
	gpr2_performance = {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}}
	for d in my_dirs:
		d_label = d.split("/")[-1].rstrip('_noisy').rstrip('_clean')
		if 'vanilla' in d_label:
			my_eps = None
		else:
			my_eps = float([z.strip(eps_token) for z in d_label.split('_') if eps_token in z][-1])
			model_loss = np.loadtxt(d+'/perfectModel_loss_clean_test.txt')
			model_t_valid = np.loadtxt(d+'/perfectModel_validity_time_clean_test.txt')
			if my_eps in model_performance['mse']:
				model_performance['mse'][my_eps] += (float(model_loss),)
				model_performance['t_valid'][my_eps] += (float(model_t_valid),)
			else:
				model_performance['mse'][my_eps] = (float(model_loss),)
				model_performance['t_valid'][my_eps] = (float(model_t_valid),)

		x_train = pd.DataFrame(np.loadtxt(d+"/loss_vec_train.txt"))
		x_test = pd.DataFrame(np.loadtxt(d+"/loss_vec_clean_test.txt"))
		gpr1_valid_test = np.loadtxt(d+'/GPR_1_prediction_validity_time_clean_test.txt')
		gpr2_valid_test = np.loadtxt(d+'/GPR_2_prediction_validity_time_clean_test.txt')
		gpr1_test = np.loadtxt(d+'/GPR_1_clean_loss_test.txt')
		gpr2_test = np.loadtxt(d+'/GPR_2_clean_loss_test.txt')

		if many_epochs:
			x_valid_test = pd.DataFrame(np.loadtxt(d+"/prediction_validity_time_clean_test.txt"))
			# x_kl_test = pd.DataFrame(np.loadtxt(d+"/kl_vec_inv_clean_test.txt"))
		if win:
			x_train = x_train.rolling(win).mean()
			x_test = x_test.rolling(win).mean()
			if many_epochs:
				x_valid_test = x_valid_test.rolling(win).mean()
				# for kk in plot_state_indices:
				# 	ax4.plot(x_kl_test.loc[:,kk].rolling(win).mean(), label=d_label)
		if my_eps is None:
			rnn_performance['mse'] += (float(np.min(x_test)),)
			# rnn_performance['mse_time'] += (float(np.nanargmin(x_test)),)
			if many_epochs:
				rnn_performance['t_valid'] += (float(np.max(x_valid_test)),)
				for tt in t_vec:
					if tt in rnn_performance['t_valid_time']:
						try:
							rnn_performance['t_valid_time'][tt] += (x_valid_test[x_valid_test.iloc[:,0]>tt].index[0],)
						except:
							rnn_performance['t_valid_time'][tt] += (np.inf,)
					else:
						try:
							rnn_performance['t_valid_time'][tt] = (x_valid_test[x_valid_test.iloc[:,0]>tt].index[0],)
						except:
							rnn_performance['t_valid_time'][tt] = (np.inf,)
		elif my_eps in hybrid_performance['mse']:
			hybrid_performance['mse'][my_eps] += (float(np.min(x_test)),)
			gpr1_performance['mse'][my_eps] += (float(np.min(gpr1_test)),)
			gpr2_performance['mse'][my_eps] += (float(np.min(gpr2_test)),)
			# for tt in t_vec:
			# 	hybrid_performance['mse_time'][my_eps][tt] += (x_test[x_test.iloc[:,0]>tt].index[0],)
			if many_epochs:
				hybrid_performance['t_valid'][my_eps] += (float(np.max(x_valid_test)),)
				gpr1_performance['t_valid'][my_eps] += (float(np.max(gpr1_valid_test)),)
				gpr2_performance['t_valid'][my_eps] += (float(np.max(gpr2_valid_test)),)
				for tt in t_vec:
					try:
						hybrid_performance['t_valid_time'][my_eps][tt] += (x_valid_test[x_valid_test.iloc[:,0]>tt].index[0],)
						gpr1_performance['t_valid_time'][my_eps][tt] += (gpr1_valid_test[gpr1_valid_test.iloc[:,0]>tt].index[0],)
						gpr2_performance['t_valid_time'][my_eps][tt] += (gpr2_valid_test[gpr2_valid_test.iloc[:,0]>tt].index[0],)
					except:
						hybrid_performance['t_valid_time'][my_eps][tt] += (np.inf,)
						gpr1_performance['t_valid_time'][my_eps][tt] += (np.inf,)
						gpr2_performance['t_valid_time'][my_eps][tt] += (np.inf,)
		else:
			hybrid_performance['mse'][my_eps] = (float(np.min(x_test)),)
			gpr1_performance['mse'][my_eps] = (float(np.min(gpr1_test)),)
			gpr2_performance['mse'][my_eps] = (float(np.min(gpr2_test)),)
			# hybrid_performance['mse_time'][my_eps] = {}
			# for tt in t_vec:
			# 	hybrid_performance['mse_time'][my_eps][tt] = (x_test[x_test.iloc[:,0]>tt].index[0],)
			if many_epochs:
				hybrid_performance['t_valid'][my_eps] = (float(np.max(x_valid_test)),)
				gpr1_performance['t_valid'][my_eps] = (float(np.max(gpr1_valid_test)),)
				gpr2_performance['t_valid'][my_eps] = (float(np.max(gpr2_valid_test)),)

				hybrid_performance['t_valid_time'][my_eps] = {}
				gpr1_performance['t_valid_time'][my_eps] = {}
				gpr2_performance['t_valid_time'][my_eps] = {}
				for tt in t_vec:
					try:
						hybrid_performance['t_valid_time'][my_eps][tt] = (x_valid_test[x_valid_test.iloc[:,0]>tt].index[0],)
						gpr1_performance['t_valid_time'][my_eps][tt] = (gpr1_valid_test[gpr1_valid_test.iloc[:,0]>tt].index[0],)
						gpr2_performance['t_valid_time'][my_eps][tt] = (gpr2_valid_test[gpr2_valid_test.iloc[:,0]>tt].index[0],)
					except:
						hybrid_performance['t_valid_time'][my_eps][tt] = (np.inf,)
						gpr1_performance['t_valid_time'][my_eps][tt] = (np.inf,)
						gpr2_performance['t_valid_time'][my_eps][tt] = (np.inf,)

	# now summarize
	test_loss_mins = {key: np.min(hybrid_performance['mse'][key]) for key in hybrid_performance['mse']}
	test_loss_maxes = {key: np.max(hybrid_performance['mse'][key]) for key in hybrid_performance['mse']}
	test_loss_means = {key: np.mean(hybrid_performance['mse'][key]) for key in hybrid_performance['mse']}
	test_loss_medians = {key: np.median(hybrid_performance['mse'][key]) for key in hybrid_performance['mse']}
	test_loss_stds = {key: np.std(hybrid_performance['mse'][key]) for key in hybrid_performance['mse']}

	gpr1_test_loss_mins = {key: np.min(gpr1_performance['mse'][key]) for key in gpr1_performance['mse']}
	gpr1_test_loss_maxes = {key: np.max(gpr1_performance['mse'][key]) for key in gpr1_performance['mse']}
	gpr1_test_loss_means = {key: np.mean(gpr1_performance['mse'][key]) for key in gpr1_performance['mse']}
	gpr1_test_loss_medians = {key: np.median(gpr1_performance['mse'][key]) for key in gpr1_performance['mse']}
	gpr1_test_loss_stds = {key: np.std(gpr1_performance['mse'][key]) for key in gpr1_performance['mse']}

	gpr2_test_loss_mins = {key: np.min(gpr2_performance['mse'][key]) for key in gpr2_performance['mse']}
	gpr2_test_loss_maxes = {key: np.max(gpr2_performance['mse'][key]) for key in gpr2_performance['mse']}
	gpr2_test_loss_means = {key: np.mean(gpr2_performance['mse'][key]) for key in gpr2_performance['mse']}
	gpr2_test_loss_medians = {key: np.median(gpr2_performance['mse'][key]) for key in gpr2_performance['mse']}
	gpr2_test_loss_stds = {key: np.std(gpr2_performance['mse'][key]) for key in gpr2_performance['mse']}

	rnn_test_loss_mins = np.min(rnn_performance['mse'])
	rnn_test_loss_maxes = np.max(rnn_performance['mse'])
	rnn_test_loss_means = np.mean(rnn_performance['mse'])
	rnn_test_loss_medians = np.median(rnn_performance['mse'])
	rnn_test_loss_stds = np.std(rnn_performance['mse'])

	ode_test_loss_mins = {key: np.min(model_performance['mse'][key]) for key in model_performance['mse']}
	ode_test_loss_maxes = {key: np.max(model_performance['mse'][key]) for key in model_performance['mse']}
	ode_test_loss_means = {key: np.mean(model_performance['mse'][key]) for key in model_performance['mse']}
	ode_test_loss_medians = {key: np.median(model_performance['mse'][key]) for key in model_performance['mse']}
	ode_test_loss_stds = {key: np.std(model_performance['mse'][key]) for key in model_performance['mse']}

	# plot summary
	fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,
		figsize = [10, 10],
		sharey=False, sharex=False)

	eps_vec = sorted(test_loss_medians.keys())

	# hybrid RNN
	median_vec = [test_loss_medians[eps] for eps in eps_vec]
	std_vec = [test_loss_stds[eps] for eps in eps_vec]
	ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid RNN', color='blue')

	# hybrid GPR 1
	median_vec = [gpr1_test_loss_medians[eps] for eps in eps_vec]
	std_vec = [gpr1_test_loss_stds[eps] for eps in eps_vec]
	ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid GPR 1', color='green', linestyle=':')

	# hybrid GPR 2
	median_vec = [gpr2_test_loss_medians[eps] for eps in eps_vec]
	std_vec = [gpr2_test_loss_stds[eps] for eps in eps_vec]
	ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid GPR 2', color='green', linestyle='--')


	ax1.set_xlabel('epsilon Model Error')
	ax1.set_ylabel('Test Loss (MSE)')

	try:
		ax1.errorbar(x=eps_vec, y=[rnn_test_loss_medians]*len(eps_vec), yerr=[rnn_test_loss_stds]*len(eps_vec) ,label='vanilla RNN', color='black')
	except:
		pass

	try:
		median_vec = [ode_test_loss_medians[eps] for eps in eps_vec]
		std_vec = [ode_test_loss_stds[eps] for eps in eps_vec]
		ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='model-only', color='red')
	except:
		pass

	ax1.legend()

	if many_epochs:
		t_valid_mins = {key: np.min(hybrid_performance['t_valid'][key]) for key in hybrid_performance['t_valid']}
		t_valid_maxes = {key: np.max(hybrid_performance['t_valid'][key]) for key in hybrid_performance['t_valid']}
		t_valid_medians = {key: np.median(hybrid_performance['t_valid'][key]) for key in hybrid_performance['t_valid']}
		t_valid_means = {key: np.mean(hybrid_performance['t_valid'][key]) for key in hybrid_performance['t_valid']}
		t_valid_stds = {key: np.std(hybrid_performance['t_valid'][key]) for key in hybrid_performance['t_valid']}

		gpr1_t_valid_mins = {key: np.min(gpr1_performance['t_valid'][key]) for key in gpr1_performance['t_valid']}
		gpr1_t_valid_maxes = {key: np.max(gpr1_performance['t_valid'][key]) for key in gpr1_performance['t_valid']}
		gpr1_t_valid_medians = {key: np.median(gpr1_performance['t_valid'][key]) for key in gpr1_performance['t_valid']}
		gpr1_t_valid_means = {key: np.mean(gpr1_performance['t_valid'][key]) for key in gpr1_performance['t_valid']}
		gpr1_t_valid_stds = {key: np.std(gpr1_performance['t_valid'][key]) for key in gpr1_performance['t_valid']}

		gpr2_t_valid_mins = {key: np.min(gpr2_performance['t_valid'][key]) for key in gpr2_performance['t_valid']}
		gpr2_t_valid_maxes = {key: np.max(gpr2_performance['t_valid'][key]) for key in gpr2_performance['t_valid']}
		gpr2_t_valid_medians = {key: np.median(gpr2_performance['t_valid'][key]) for key in gpr2_performance['t_valid']}
		gpr2_t_valid_means = {key: np.mean(gpr2_performance['t_valid'][key]) for key in gpr2_performance['t_valid']}
		gpr2_t_valid_stds = {key: np.std(gpr2_performance['t_valid'][key]) for key in gpr2_performance['t_valid']}

		rnn_t_valid_mins = np.min(rnn_performance['t_valid'])
		rnn_t_valid_maxes = np.max(rnn_performance['t_valid'])
		rnn_t_valid_means = np.mean(rnn_performance['t_valid'])
		rnn_t_valid_medians = np.median(rnn_performance['t_valid'])
		rnn_t_valid_stds = np.std(rnn_performance['t_valid'])

		ode_t_valid_mins = {key: np.min(model_performance['t_valid'][key]) for key in model_performance['t_valid']}
		ode_t_valid_maxes = {key: np.max(model_performance['t_valid'][key]) for key in model_performance['t_valid']}
		ode_t_valid_means = {key: np.mean(model_performance['t_valid'][key]) for key in model_performance['t_valid']}
		ode_t_valid_medians = {key: np.median(model_performance['t_valid'][key]) for key in model_performance['t_valid']}
		ode_t_valid_stds = {key: np.std(model_performance['t_valid'][key]) for key in model_performance['t_valid']}

		# hybrid RNN
		eps_vec = sorted(t_valid_medians.keys())
		median_vec = [t_valid_medians[eps] for eps in eps_vec]
		std_vec = [t_valid_stds[eps] for eps in eps_vec]
		ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid RNN', color='blue')

		# hybrid GPR 1
		eps_vec = sorted(gpr1_t_valid_medians.keys())
		median_vec = [gpr1_t_valid_medians[eps] for eps in eps_vec]
		std_vec = [gpr1_t_valid_stds[eps] for eps in eps_vec]
		ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid GPR 1', color='green', linestyle=':')

		# hybrid GPR 2
		eps_vec = sorted(gpr2_t_valid_medians.keys())
		median_vec = [gpr2_t_valid_medians[eps] for eps in eps_vec]
		std_vec = [gpr2_t_valid_stds[eps] for eps in eps_vec]
		ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid GPR 2', color='green', linestyle='--')

		ax2.set_xlabel('epsilon Model Error')
		ax2.set_ylabel('Validity Time')

		try:
			ax2.errorbar(x=eps_vec, y=[rnn_t_valid_medians]*len(eps_vec), yerr=[rnn_t_valid_stds]*len(eps_vec) ,label='vanilla RNN', color='black')
		except:
			pass

		try:
			special_eps_vec = [foo for foo in eps_vec if foo!=0]
			median_vec = [ode_t_valid_medians[eps] for eps in special_eps_vec]
			std_vec = [ode_t_valid_stds[eps] for eps in special_eps_vec]
			ax2.errorbar(x=special_eps_vec, y=median_vec, yerr=std_vec, label='model-only', color='red')
		except:
			pass
		ax2.legend()

	fig.suptitle('Performance on Test Set Under Varying Model Error')
	fig.savefig(fname=output_fname)
	plt.close(fig)

	# TRAIN TIME ANALYSIS
	# plot summary
	colors = ['red','orange','green','blue','purple','black']
	fig, ax2 = plt.subplots(nrows=1, ncols=1,
		figsize = [10, 10],
		sharey=False, sharex=False)
	for i_tt in range(len(t_vec)):
		tt = t_vec[i_tt]
		if many_epochs:
			try:
				t_valid_mins = {key: np.min(hybrid_performance['t_valid_time'][key][tt]) for key in hybrid_performance['t_valid_time']}
			except:
				pdb.set_trace()
			t_valid_maxes = {key: np.max(hybrid_performance['t_valid_time'][key][tt]) for key in hybrid_performance['t_valid_time']}
			t_valid_medians = {key: np.median(hybrid_performance['t_valid_time'][key][tt]) for key in hybrid_performance['t_valid_time']}
			t_valid_means = {key: np.mean(hybrid_performance['t_valid_time'][key][tt]) for key in hybrid_performance['t_valid_time']}
			t_valid_stds = {key: np.std(hybrid_performance['t_valid_time'][key][tt]) for key in hybrid_performance['t_valid_time']}

			rnn_t_valid_mins = np.min(rnn_performance['t_valid_time'][tt])
			rnn_t_valid_maxes = np.max(rnn_performance['t_valid_time'][tt])
			rnn_t_valid_means = np.mean(rnn_performance['t_valid_time'][tt])
			rnn_t_valid_medians = np.median(rnn_performance['t_valid_time'][tt])
			rnn_t_valid_stds = np.std(rnn_performance['t_valid_time'][tt])

			eps_vec = sorted(t_valid_medians.keys())
			median_vec = [t_valid_medians[eps] for eps in eps_vec]
			std_vec = [t_valid_stds[eps] for eps in eps_vec]

			ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid RNN (t>{0})'.format(tt), linestyle='-', color=colors[i_tt])
			ax2.set_xlabel('epsilon Model Error')
			ax2.set_ylabel('Train Time (opt Validity Time)')

			try:
				ax2.errorbar(x=eps_vec, y=[rnn_t_valid_medians]*len(eps_vec), yerr=[rnn_t_valid_stds]*len(eps_vec), linestyle=':', label='vanilla RNN (t>{0})'.format(tt), color=colors[i_tt])
			except:
				pass

			# try:
			# 	special_eps_vec = [foo for foo in eps_vec if foo!=0]
			# 	median_vec = [ode_t_valid_medians[eps] for eps in special_eps_vec]
			# 	std_vec = [ode_t_valid_stds[eps] for eps in special_eps_vec]
			# 	ax2.errorbar(x=special_eps_vec, y=median_vec, yerr=std_vec, label='model-only', color='red')
			# except:
			# 	pass

	ax2.legend()
	ax2.set_yscale('log')
	fig.suptitle('Train-time until reaching optimal test performance (w.r.t. model error)')
	fig.savefig(fname=output_fname+'_train_time')
	plt.close(fig)


def extract_hidden_size_performance(my_dirs, output_fname="./hidden_size_comparisons", win=1000, many_epochs=True, hs_token='hs'):

	# first, get sizes of things...max window size is 10% of whole test set.
	d_label = my_dirs[0].split("/")[-1].rstrip('_noisy').rstrip('_clean')
	x_test = pd.DataFrame(np.loadtxt(my_dirs[0]+"/loss_vec_clean_test.txt"))
	n_vals = len(x_test)

	win = min(win,n_vals//3)
	model_performance = {'mse':(), 't_valid':()}
	rnn_performance = {'mse':{}, 'mse_time':{}, 't_valid':{}, 't_valid_time':{}}
	hybrid_performance = {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}}
	dict_performance = {'rnn':rnn_performance, 'hybrid': hybrid_performance, 'model':model_performance}
	for d in my_dirs:
		d_label = d.split("/")[-1].rstrip('_noisy').rstrip('_clean')
		my_hs = float([z.strip(hs_token) for z in d_label.split('_') if hs_token in z][-1])
		try:
			model_loss = np.loadtxt(d+'/perfectModel_loss_test.txt')
			model_t_valid = np.loadtxt(d+'/perfectModel_validity_time_test.txt')
			dict_performance['model']['mse'] += (float(model_loss),)
			dict_performance['model']['t_valid'] += (float(model_t_valid),)
		except:
			pass

		x_train = pd.DataFrame(np.loadtxt(d+"/loss_vec_train.txt"))
		x_test = pd.DataFrame(np.loadtxt(d+"/loss_vec_clean_test.txt"))
		if many_epochs:
			x_valid_test = pd.DataFrame(np.loadtxt(d+"/prediction_validity_time_clean_test.txt"))
			# x_kl_test = pd.DataFrame(np.loadtxt(d+"/kl_vec_inv_clean_test.txt"))
		if win:
			x_train = x_train.rolling(win).mean()
			x_test = x_test.rolling(win).mean()
			if many_epochs:
				x_valid_test = x_valid_test.rolling(win).mean()
				# for kk in plot_state_indices:
				# 	ax4.plot(x_kl_test.loc[:,kk].rolling(win).mean(), label=d_label)
		# if my_hs is None:
		# 	rnn_performance['mse'] += (float(np.min(x_test)),)
		# 	if many_epochs:
		# 		rnn_performance['t_valid'] += (float(np.max(x_valid_test)),)
		if 'vanilla' in d_label:
			mtype = 'rnn'
		elif 'mech' in d_label:
			mtype = 'hybrid'
		else:
			pass

		if my_hs in dict_performance[mtype]['mse']:
			dict_performance[mtype]['mse'][my_hs] += (float(np.min(x_test)),)
			dict_performance[mtype]['mse_time'][my_hs] += (float(np.nanargmin(x_test)),)
			if many_epochs:
				dict_performance[mtype]['t_valid'][my_hs] += (float(np.max(x_valid_test)),)
				dict_performance[mtype]['t_valid_time'][my_hs] += (float(np.nanargmax(x_valid_test)),)
		else:
			dict_performance[mtype]['mse'][my_hs] = (float(np.min(x_test)),)
			dict_performance[mtype]['mse_time'][my_hs] = (float(np.nanargmin(x_test)),)
			if many_epochs:
				dict_performance[mtype]['t_valid'][my_hs] = (float(np.max(x_valid_test)),)
				dict_performance[mtype]['t_valid_time'][my_hs] = (float(np.nanargmax(x_valid_test)),)

	# now summarize
	test_loss_mins = {key: np.min(dict_performance['hybrid']['mse'][key]) for key in dict_performance['hybrid']['mse']}
	test_loss_maxes = {key: np.max(dict_performance['hybrid']['mse'][key]) for key in dict_performance['hybrid']['mse']}
	test_loss_means = {key: np.mean(dict_performance['hybrid']['mse'][key]) for key in dict_performance['hybrid']['mse']}
	test_loss_medians = {key: np.median(dict_performance['hybrid']['mse'][key]) for key in dict_performance['hybrid']['mse']}
	test_loss_stds = {key: np.std(dict_performance['hybrid']['mse'][key]) for key in dict_performance['hybrid']['mse']}
	rnn_test_loss_mins = {key: np.min(dict_performance['rnn']['mse'][key]) for key in dict_performance['rnn']['mse']}
	rnn_test_loss_maxes = {key: np.max(dict_performance['rnn']['mse'][key]) for key in dict_performance['rnn']['mse']}
	rnn_test_loss_means = {key: np.mean(dict_performance['rnn']['mse'][key]) for key in dict_performance['rnn']['mse']}
	rnn_test_loss_medians = {key: np.median(dict_performance['rnn']['mse'][key]) for key in dict_performance['rnn']['mse']}
	rnn_test_loss_stds = {key: np.std(dict_performance['rnn']['mse'][key]) for key in dict_performance['rnn']['mse']}
	ode_test_loss_mins = np.min(dict_performance['model']['mse'])
	ode_test_loss_maxes = np.max(dict_performance['model']['mse'])
	ode_test_loss_means = np.mean(dict_performance['model']['mse'])
	ode_test_loss_medians = np.median(dict_performance['model']['mse'])
	ode_test_loss_stds = np.std(dict_performance['model']['mse'])


	# plot summary
	fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,
		figsize = [10, 10],
		sharey=False, sharex=False)

	eps_vec = sorted(test_loss_medians.keys())
	median_vec = [test_loss_medians[eps] for eps in eps_vec]
	std_vec = [test_loss_stds[eps] for eps in eps_vec]

	ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid RNN (eps=0.05)', color='blue')
	ax1.set_xlabel('Dimension of Hidden State')
	ax1.set_ylabel('Test Loss (logMSE)')

	eps_vec = sorted(rnn_test_loss_medians.keys())
	median_vec = [rnn_test_loss_medians[eps] for eps in eps_vec]
	std_vec = [rnn_test_loss_stds[eps] for eps in eps_vec]
	ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='vanilla RNN', color='black')

	try:
		ax1.errorbar(x=eps_vec, y=[ode_test_loss_medians]*len(eps_vec), yerr=[ode_test_loss_stds]*len(eps_vec), label='model-only (eps=0.05)', color='red')
	except:
		pass

	ax1.set_yscale('log')
	ax1.legend()

	if many_epochs:
		t_valid_mins = {key: np.min(dict_performance['hybrid']['t_valid'][key]) for key in dict_performance['hybrid']['t_valid']}
		t_valid_maxes = {key: np.max(dict_performance['hybrid']['t_valid'][key]) for key in dict_performance['hybrid']['t_valid']}
		t_valid_means = {key: np.mean(dict_performance['hybrid']['t_valid'][key]) for key in dict_performance['hybrid']['t_valid']}
		t_valid_medians = {key: np.median(dict_performance['hybrid']['t_valid'][key]) for key in dict_performance['hybrid']['t_valid']}
		t_valid_stds = {key: np.std(dict_performance['hybrid']['t_valid'][key]) for key in dict_performance['hybrid']['t_valid']}

		rnn_t_valid_mins = {key: np.min(dict_performance['rnn']['t_valid'][key]) for key in dict_performance['rnn']['t_valid']}
		rnn_t_valid_maxes = {key: np.max(dict_performance['rnn']['t_valid'][key]) for key in dict_performance['rnn']['t_valid']}
		rnn_t_valid_means = {key: np.mean(dict_performance['rnn']['t_valid'][key]) for key in dict_performance['rnn']['t_valid']}
		rnn_t_valid_medians = {key: np.median(dict_performance['rnn']['t_valid'][key]) for key in dict_performance['rnn']['t_valid']}
		rnn_t_valid_stds = {key: np.std(dict_performance['rnn']['t_valid'][key]) for key in dict_performance['rnn']['t_valid']}

		ode_t_valid_mins = np.min(dict_performance['model']['t_valid'])
		ode_t_valid_maxes = np.max(dict_performance['model']['t_valid'])
		ode_t_valid_means = np.mean(dict_performance['model']['t_valid'])
		ode_t_valid_medians = np.median(dict_performance['model']['t_valid'])
		ode_t_valid_stds = np.std(dict_performance['model']['t_valid'])

		eps_vec = sorted(t_valid_medians.keys())
		median_vec = [t_valid_medians[eps] for eps in eps_vec]
		std_vec = [t_valid_stds[eps] for eps in eps_vec]

		ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid RNN (eps=0.05)', color='blue')
		ax2.set_xlabel('Dimension of Hidden State')
		ax2.set_ylabel('Validity Time')

		eps_vec = sorted(rnn_t_valid_medians.keys())
		median_vec = [rnn_t_valid_medians[eps] for eps in eps_vec]
		std_vec = [rnn_t_valid_stds[eps] for eps in eps_vec]
		ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='vanilla RNN', color='black')

		try:
			ax2.errorbar(x=eps_vec, y=[ode_t_valid_medians]*len(eps_vec), yerr=[ode_t_valid_stds]*len(eps_vec) ,label='model-only (eps=0.05)', color='red')
		except:
			pass

		ax2.legend()

	fig.suptitle('Performance on Test Set Under Varying Size of Hidden Dimension')
	fig.savefig(fname=output_fname)
	plt.close(fig)

	# plot summary of Time to Train
	test_loss_mins = {key: np.min(dict_performance['hybrid']['mse_time'][key]) for key in dict_performance['hybrid']['mse_time']}
	test_loss_maxes = {key: np.max(dict_performance['hybrid']['mse_time'][key]) for key in dict_performance['hybrid']['mse_time']}
	test_loss_means = {key: np.mean(dict_performance['hybrid']['mse_time'][key]) for key in dict_performance['hybrid']['mse_time']}
	test_loss_medians = {key: np.median(dict_performance['hybrid']['mse_time'][key]) for key in dict_performance['hybrid']['mse_time']}
	test_loss_stds = {key: np.std(dict_performance['hybrid']['mse_time'][key]) for key in dict_performance['hybrid']['mse_time']}
	rnn_test_loss_mins = {key: np.min(dict_performance['rnn']['mse_time'][key]) for key in dict_performance['rnn']['mse_time']}
	rnn_test_loss_maxes = {key: np.max(dict_performance['rnn']['mse_time'][key]) for key in dict_performance['rnn']['mse_time']}
	rnn_test_loss_means = {key: np.mean(dict_performance['rnn']['mse_time'][key]) for key in dict_performance['rnn']['mse_time']}
	rnn_test_loss_medians = {key: np.median(dict_performance['rnn']['mse_time'][key]) for key in dict_performance['rnn']['mse_time']}
	rnn_test_loss_stds = {key: np.std(dict_performance['rnn']['mse_time'][key]) for key in dict_performance['rnn']['mse_time']}
	# ode_test_loss_mins = np.min(dict_performance['model']['mse_time'])
	# ode_test_loss_maxes = np.max(dict_performance['model']['mse_time'])
	# ode_test_loss_means = np.mean(dict_performance['model']['mse_time'])
	# ode_test_loss_medians = np.median(dict_performance['model']['mse_time'])
	# ode_test_loss_stds = np.std(dict_performance['model']['mse_time'])


	fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,
		figsize = [10, 10],
		sharey=False, sharex=False)

	eps_vec = sorted(test_loss_medians.keys())
	median_vec = [test_loss_medians[eps] for eps in eps_vec]
	std_vec = [test_loss_stds[eps] for eps in eps_vec]

	ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid RNN (eps=0.05)', color='blue')
	ax1.set_xlabel('Dimension of Hidden State')
	ax1.set_ylabel('Train time (opt MSE)')

	eps_vec = sorted(rnn_test_loss_medians.keys())
	median_vec = [rnn_test_loss_medians[eps] for eps in eps_vec]
	std_vec = [rnn_test_loss_stds[eps] for eps in eps_vec]
	ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='vanilla RNN', color='black')

	# try:
	# 	ax1.errorbar(x=eps_vec, y=[ode_test_loss_medians]*len(eps_vec), yerr=[ode_test_loss_stds]*len(eps_vec), label='model-only (eps=0.05)', color='red')
	# except:
	# 	pass

	# ax1.set_yscale('log')
	ax1.legend()

	if many_epochs:
		t_valid_mins = {key: np.min(dict_performance['hybrid']['t_valid_time'][key]) for key in dict_performance['hybrid']['t_valid_time']}
		t_valid_maxes = {key: np.max(dict_performance['hybrid']['t_valid_time'][key]) for key in dict_performance['hybrid']['t_valid_time']}
		t_valid_means = {key: np.mean(dict_performance['hybrid']['t_valid_time'][key]) for key in dict_performance['hybrid']['t_valid_time']}
		t_valid_medians = {key: np.median(dict_performance['hybrid']['t_valid_time'][key]) for key in dict_performance['hybrid']['t_valid_time']}
		t_valid_stds = {key: np.std(dict_performance['hybrid']['t_valid_time'][key]) for key in dict_performance['hybrid']['t_valid_time']}

		rnn_t_valid_mins = {key: np.min(dict_performance['rnn']['t_valid_time'][key]) for key in dict_performance['rnn']['t_valid_time']}
		rnn_t_valid_maxes = {key: np.max(dict_performance['rnn']['t_valid_time'][key]) for key in dict_performance['rnn']['t_valid_time']}
		rnn_t_valid_means = {key: np.mean(dict_performance['rnn']['t_valid_time'][key]) for key in dict_performance['rnn']['t_valid_time']}
		rnn_t_valid_medians = {key: np.median(dict_performance['rnn']['t_valid_time'][key]) for key in dict_performance['rnn']['t_valid_time']}
		rnn_t_valid_stds = {key: np.std(dict_performance['rnn']['t_valid_time'][key]) for key in dict_performance['rnn']['t_valid_time']}

		# ode_t_valid_mins = np.min(dict_performance['model']['t_valid_time'])
		# ode_t_valid_maxes = np.max(dict_performance['model']['t_valid_time'])
		# ode_t_valid_means = np.mean(dict_performance['model']['t_valid_time'])
		# ode_t_valid_medians = np.median(dict_performance['model']['t_valid_time'])
		# ode_t_valid_stds = np.std(dict_performance['model']['t_valid_time'])

		eps_vec = sorted(t_valid_medians.keys())
		median_vec = [t_valid_medians[eps] for eps in eps_vec]
		std_vec = [t_valid_stds[eps] for eps in eps_vec]

		ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid RNN (eps=0.05)', color='blue')
		ax2.set_xlabel('Dimension of Hidden State')
		ax2.set_ylabel('Train time (opt Validity Time)')

		eps_vec = sorted(rnn_t_valid_medians.keys())
		median_vec = [rnn_t_valid_medians[eps] for eps in eps_vec]
		std_vec = [rnn_t_valid_stds[eps] for eps in eps_vec]
		ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='vanilla RNN', color='black')

		# try:
		# 	ax2.errorbar(x=eps_vec, y=[ode_t_valid_medians]*len(eps_vec), yerr=[ode_t_valid_stds]*len(eps_vec) ,label='model-only (eps=0.05)', color='red')
		# except:
		# 	pass

		ax2.legend()

	fig.suptitle('Train-time until reaching optimal test performance (w.r.t. hidden size)')
	fig.savefig(fname=output_fname+'_train_time')
	plt.close(fig)
