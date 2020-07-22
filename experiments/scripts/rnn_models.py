import os, sys
import warnings
import numpy as np
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from utils import phase_plot, traj_div_time, train_chaosRNN, forward_chaos_hybrid_full, forward_chaos_pureML
from line_profiler import LineProfiler
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pdb


def normalize(y, norm_dict):
	return (y - norm_dict['Ymin']) / (norm_dict['Ymax'] - norm_dict['Ymin'])

def unnormalize(y_norm, norm_dict):
	return norm_dict['Ymin'] + y_norm * (norm_dict['Ymax'] - norm_dict['Ymin'])

def setup_RNN(setts, training_fname, testing_fname, odeInst, profile=False):
	t0 = time()

	if setts['omit_z']:
		keep_inds = np.arange(odeInst.K-1)
	else:
		keep_inds = np.arange(odeInst.K)
		setts['plot_state_indices'] = odeInst.plot_state_indices()


	# read TRAIN data
	train_set = np.load(training_fname)
	y_clean_train = train_set['X_train'][:,keep_inds]
	y_noisy_train = train_set['X_train'][:,keep_inds]



	normz_info = {
				'Ymax': np.max(y_noisy_train, axis=0),
				'Ymin': np.min(y_noisy_train, axis=0),
				'Ymean': np.mean(y_noisy_train, axis=0),
				'Ysd': np.std(y_noisy_train, axis=0)
				}

	setts['normz_info'] = normz_info

	setts['y_clean_train'] = normalize(norm_dict=normz_info, y=y_clean_train)
	setts['y_noisy_train'] = normalize(norm_dict=normz_info, y=y_noisy_train)
	# setts['y_clean_trainSynch'] = normalize(norm_dict=normz_info, y=train_set['y_clean_synch'])
	# setts['y_noisy_trainSynch'] = normalize(norm_dict=normz_info, y=train_set['y_noisy_synch'])
	setts['y_fast_train'] = train_set['y_fast']

	# read and normalize TEST data
	test_set = np.load(testing_fname)
	setts['y_clean_test_long'] = normalize(norm_dict=normz_info, y=test_set['X_test'][:,keep_inds])
	y_clean_test = []
	y_noisy_test = []
	y_clean_testSynch = []
	y_noisy_testSynch = []
	for c in range(test_set['X_test_traj'].shape[0]):
		y_clean_test.append(normalize(norm_dict=normz_info, y=test_set['X_test_traj'][c,:,keep_inds].transpose()))
		y_noisy_test.append(normalize(norm_dict=normz_info, y=test_set['X_test_traj'][c,:,keep_inds].transpose()))
		y_clean_testSynch.append(normalize(norm_dict=normz_info, y=test_set['X_test_traj_synch'][c,:,keep_inds].transpose()))
		y_noisy_testSynch.append(normalize(norm_dict=normz_info, y=test_set['X_test_traj_synch'][c,:,keep_inds].transpose()))

	setts['y_clean_test'] = np.stack(y_clean_test)
	setts['y_noisy_test'] = np.stack(y_noisy_test)
	setts['y_clean_testSynch'] = np.stack(y_clean_testSynch)
	setts['y_noisy_testSynch'] = np.stack(y_noisy_testSynch)

	# get state names
	setts['model_params']['state_names'] = odeInst.get_state_names()

	setts['model'] = odeInst.rhs

	setts['ODE'] = odeInst

	# setts['mode'] = 'original'
	setts['learn_residuals'] = setts['use_physics_as_bias']
	if setts['use_physics_as_bias']:
		setts['forward'] = forward_chaos_hybrid_full
	else:
		setts['forward'] = forward_chaos_pureML

	if profile:
		lp = LineProfiler()
		# lp.add_function(forward_chaos_pureML)
		lp.add_function(RNN.forward)
		if setts['old']:
			lp_wrapper = lp(train_chaosRNN)
		else:
			lp_wrapper = lp(train_RNN_new)
		lp_wrapper(**setts)
		lp.print_stats()
	else:
		if setts['old']:
			train_chaosRNN(**setts)
		else:
			train_RNN_new(**setts)

	print('Ran training in:', time()-t0)
	return


class RNN(nn.Module):
	def __init__(self,
			input_size,
			hidden_size=50,
			output_size=None,
			cell_type='RNN',
			embed_physics_prediction=False,
			use_physics_as_bias=False,
			dtype=torch.float,
			t_synch=1000,
			teacher_force_probability=0.0,
			norm_dict=None,
			ode_params=None,
			ode=None,
			output_path='default_output',
			max_plot=None,
			mode=None, #'original'
			use_manual_seed=False,
			component_wise=False,
			do_euler=True,
			hidden_euler=False):

		super().__init__()
		self.t0_overall = time()
		self.t0_epoch = time()
		if output_size is None:
			output_size = input_size
		self.exchangeable_states = ode.exchangeable_states
		self.do_euler = do_euler
		self.hidden_euler = hidden_euler
		self.component_wise = component_wise
		self.cell_type = cell_type
		self.use_manual_seed = use_manual_seed
		self.mode = mode
		self.output_path = output_path
		self.teacher_force_probability = teacher_force_probability
		self.t_synch = t_synch
		self.dtype = dtype
		self.hidden_size = hidden_size

		if self.component_wise:
			self.n_components = output_size
			self.input_size = 1
			self.output_size = 1
		else:
			self.n_components = 1
			self.input_size = input_size
			self.output_size = output_size
		self.embed_physics_prediction = embed_physics_prediction
		self.use_physics_as_bias = use_physics_as_bias
		self.ode_params = {'method': ode_params['ode_int_method'],
							'atol': ode_params['ode_int_atol'],
							'rtol': ode_params['ode_int_rtol'],
							'max_step': ode_params['ode_int_max_step']}
		self.time_avg_norm = ode_params['time_avg_norm']
		self.delta_t = ode_params['delta_t']
		self.tspan = [0, self.delta_t]
		self.t_eval = np.array([self.delta_t])
		if max_plot is None:
			self.max_plot = int(np.floor(30./self.delta_t))
		else:
			self.max_plot = max_plot

		self.ode = ode

		for key in norm_dict:
			norm_dict[key] = torch.FloatTensor(norm_dict[key])
		self.norm_dict = norm_dict

		if self.embed_physics_prediction:
			self.input_size = 2*self.input_size

		self.cum_time = []

		#This is how to add a parameter
		# self.w = nn.Parameter(scalar(0.1), requires_grad=True)

		# Default is RNN w/ ReLU
		self.lookup = {}
		if cell_type=='LSTM':
			self.cell = nn.LSTMCell(self.input_size, self.hidden_size)
			self.use_c_cell = True
		elif cell_type=='GRU':
			self.cell = nn.GRUCell(self.input_size, self.hidden_size)
			self.use_c_cell = False
		else:
			self.use_c_cell = False
			self.cell = nn.RNNCell(self.input_size, self.hidden_size, nonlinearity='relu')
			self.lookup = {'cell.weight_hh': 'A_mat',
						'cell.weight_ih': 'B_mat',
						'hidden2pred.weight': 'C_mat',
						'cell.bias_hh': 'a_vec',
						'hidden2pred.bias': 'b_vec'}


		# The linear layer that maps from hidden state space to tag space
		self.hidden2pred = nn.Linear(self.hidden_size, self.output_size)

		# maybe do manual weight initialization?
		self.initialize_weights()

		# Remember weights
		self.weight_history = {}
		self.remember_weights()

		# Create empty hidden states
		self.clear_hidden()

	def initialize_weights(self):
		if self.mode=='original':
			for name, val in self.named_parameters(): #self.state_dict():
				if self.use_manual_seed:
					torch.manual_seed(0)
				nn.init.normal_(val, mean=0.0, std=0.1)
			# CUSTOM slight adjustment to parameterization
			# for some reason, pytorch includes 2 redundant bias terms in the cell
			self.cell.bias_ih.data = torch.zeros(self.cell.bias_ih.data.shape).type(self.dtype)
			self.cell.bias_ih.requires_grad = False
		elif self.mode=='fromfile':
			for name, val in self.named_parameters():
				# pdb.set_trace()
				if name in self.lookup:
					fname = os.path.join('/Users/matthewlevine/test_outputs/l63/rnn_output/10_epochs/pureRNN_vanilla_old', self.lookup[name] + '.txt')
					val.data = torch.FloatTensor(np.loadtxt(fname=fname)).type(self.dtype)
		else:
			if self.use_manual_seed:
				torch.manual_seed(0)
			print('Using default parameter initialization from PyTorch. Godspeed!')

	def clear_hidden(self):
		self.h_t = None
		self.c_t = None

	def detach_hidden(self):
		self.h_t.detach_()
		if self.use_c_cell:
			self.c_t.detach_()

	def normalize(self, y):
		return normalize(norm_dict=self.norm_dict, y=y)

	def unnormalize(self, y_norm):
		return unnormalize(norm_dict=self.norm_dict, y_norm=y_norm)

	def get_physics_prediction(self, X):
		#input and output are unnormalized
		if X.ndim==2:
			do_squeeze=True
			X = X[:,None,:]
		else:
			do_squeeze=False

		y_pred = np.zeros(X.shape)
		for i1 in range(X.shape[0]):
			for i2 in range(X.shape[1]):
				# check if bad initial condition
				if (any(abs(X[i1,i2,:])>1000) or any(np.isnan(X[i1,i2,:]))):
					if not self.solver_failed[i1]: # only print if it is new/recent news!
						# pdb.set_trace()
						print('ODE initial conditions are huge, so not even trying to solve the system. Applying the Identity forward map instead.',X[i1,i2,:])
					self.solver_failed[i1] = True
				else:
					self.solver_failed[i1] = False

				if not self.solver_failed[i1]:
					sol = solve_ivp(fun=lambda t, y: self.ode.rhs(y, t), t_span=self.tspan, y0=X[i1,i2,:], t_eval=self.t_eval, **self.ode_params)
					# sol = solve_ivp(fun=lambda t, y: model(y, t, *model_params['ode_params']), t_span=(self.tspan[0], self.tspan[-1]), y0=y0.T, method=model_params['ode_int_method'], rtol=model_params['ode_int_rtol'], atol=model_params['ode_int_atol'], max_step=model_params['ode_int_max_step'], t_eval=self.tspan)
					y_out = sol.y.T
					if not sol.success:
						# solver failed
						print('ODE solver has failed at ic=',X[i1,i2,:])
						self.solver_failed[i1] = True

				if self.solver_failed[i1]:
					y_pred[i1,i2,:] = np.copy(X[i1,i2,:]) # persist previous solution
				else:
					# solver is OKAY--use the solution like a good boy!
					y_pred[i1,i2,:] = y_out[-1,:]

		foo_out = torch.FloatTensor(y_pred).type(self.dtype)
		if do_squeeze:
			foo_out = foo_out.squeeze(1)

		return foo_out

	def forward(self, input_state_sequence, n_steps=None, physical_prediction_sequence=None, train=True, synch_mode=False):
		# input_state_sequence should be normalized
		# physical_prediction_sequence should be normalized
		if n_steps is None:
			n_steps = input_state_sequence.shape[1]

		rnn_preds = [] #output of the RNN (i.e. residual)
		full_preds = [] #final output prediction (i.e. Psi0(x_t) + RNN(x_t,h_t))
		hidden_preds = []

		#useful link for teacher-forcing: https://towardsdatascience.com/lstm-for-time-series-prediction-de8aeb26f2ca
		if self.h_t is None:
			self.h_t = torch.zeros((self.n_components, input_state_sequence.size(0), self.hidden_size), requires_grad=True).type(self.dtype) # (batch, hidden_size)
			if train:
				nn.init.normal_(self.h_t,0.0,0.1)
		if self.c_t is None and self.use_c_cell:
			self.c_t = torch.zeros((self.n_components, input_state_sequence.size(0), self.hidden_size), requires_grad=True).type(self.dtype) # (batch, hidden_size)
			if train:
				nn.init.normal_(self.c_t,0.0,0.1)
		full_rnn_pred = input_state_sequence[:,0] #normalized

		self.solver_failed = [False for _ in range(full_rnn_pred.shape[0])]
		# consider Scheduled Sampling (https://arxiv.org/abs/1506.03099) where probability of using RNN-output increases as you train.
		for t in range(n_steps):
			# get input to hidden state
			if train:
				# consider teacher forcing (using RNN output prediction as next training input instead of training data)
				if t>self.t_synch and random.random()<self.teacher_force_probability:
					x_now = full_rnn_pred #feed RNN prediction back in as next input
				else:
					x_now = input_state_sequence[:,t,:]
			else:
				if synch_mode:
					x_now = input_state_sequence[:,t,:]
				else:
					x_now = full_rnn_pred

			x_now = x_now.view((input_state_sequence[:,0].shape))

			if self.use_physics_as_bias or self.embed_physics_prediction:
				if physical_prediction_sequence is not None:
					physics_pred = physical_prediction_sequence[:,t,:]
				else:
					if x_now.ndim==1:
						x_now = x_now[None,:]
					ic = self.unnormalize(x_now).detach().numpy()
					physics_pred = self.normalize(self.get_physics_prediction(X=ic))
			else:
				physics_pred = 0

			if self.embed_physics_prediction:
				input_t = torch.stack(x_now, physics_pred)
			else:
				input_t = x_now

			# evolve hidden state(s)
			h_t_new = self.h_t.clone()
			if self.use_c_cell: # LSTM / GRU
				c_t_new = self.c_t.clone()
				for n in range(self.n_components):
					if self.component_wise:
						cell_input = input_t[None,:,n].transpose(0,1)
					else:
						cell_input = input_t
					# pdb.set_trace()
					foo_h_n, foo_c_n = self.cell(cell_input, (self.h_t[n,:].view((self.h_t.shape[1], self.h_t.shape[2])), self.c_t[n].view((self.c_t.shape[1], self.c_t.shape[2])))) # input needs to be dim (batch, input_size)
					if self.hidden_euler:
						h_t_new[n] += self.delta_t * foo_h_n
						c_t_new[n] += self.delta_t * foo_c_n
					else:
						h_t_new[n] = foo_h_n
						c_t_new[n] = foo_c_n
				self.c_t = c_t_new
			else: # standard RNN
				for n in range(self.n_components):
					if self.component_wise:
						cell_input = input_t[None,:,n].transpose(0,1)
					else:
						cell_input = input_t

					foo_h_n = self.cell(cell_input, self.h_t[n,:].view((self.h_t.shape[1], self.h_t.shape[2])))
					if self.hidden_euler:
						h_t_new[n] += self.delta_t * foo_h_n
					else:
						h_t_new[n] = foo_h_n

			self.h_t = h_t_new
			rnn_pred = self.hidden2pred(self.h_t).view(input_t.shape[0], 1, self.input_size*self.n_components) # (batch, n_steps, input_size)
			# print(rnn_pred.shape)
			if self.use_physics_as_bias:
				rnn_pred = rnn_pred.view(physics_pred.shape)

			if self.do_euler:
				full_rnn_pred = x_now.view(rnn_pred.shape) + (self.use_physics_as_bias * physics_pred) + (self.delta_t * rnn_pred)
			else:
				full_rnn_pred = self.use_physics_as_bias * physics_pred + rnn_pred

			full_preds += [full_rnn_pred]
			rnn_preds += [rnn_pred]
			hidden_preds += [h_t_new]

		full_preds = torch.stack(full_preds, 1).squeeze(2)
		rnn_preds = torch.stack(rnn_preds, 1).squeeze(2)
		hidden_preds = torch.stack(hidden_preds, 0) #regular: (1,2,50)
		hidden_preds.transpose_(0,2)
		# hidden_preds = hidden_preds.reshape(input_t.shape[0], self.n_components, n_steps, self.hidden_size)
		# check these hidden_preds under component-wise situation

		return full_preds, rnn_preds, hidden_preds

	def make_invariant_measure_plots(self, Xtrue, Xpred, hidden_states, epoch):
		plot_dir = os.path.join(self.output_path, 'inv_state_long')
		os.makedirs(plot_dir, exist_ok=True)

		n_steps, n_states = Xtrue.shape
		if self.exchangeable_states:
			fig, ax = plt.subplots(1, 1, figsize=[12,10])
			sns.kdeplot(Xtrue.flatten(), ax=ax, label='True')
			try:
				sns.kdeplot(Xpred.flatten(), ax=ax, label='Predicted')
			except:
				print('couldnt plot predicted KDE')
			ax.legend()
			ax.set_xlabel(r'$X_k$')
		else:
			fig, ax_list = plt.subplots(1, n_states, figsize=[12,6])
			for s in range(n_states):
				ax = ax_list[s]
				sns.kdeplot(Xtrue[:,s], ax=ax, label='True')
				try:
					sns.kdeplot(Xpred[:,s], ax=ax, label='Predicted')
				except:
					print('couldnt plot predicted KDE')
				ax.legend()
				ax.set_xlabel(r'$X_{s}$'.format(s=s))
		fig.suptitle('Invariant Measure Predictions')
		fig.savefig(fname=os.path.join(plot_dir,'epoch{epoch}'.format(epoch=epoch)))
		plt.close(fig)

		# hidden-state invariant measure
		hidden_inv_dir = os.path.join(self.output_path, 'inv_hidden_long')
		os.makedirs(hidden_inv_dir, exist_ok=True)

		n_hidden_states = hidden_states.shape[-1]
		fig_h, ax_h = plt.subplots(1, 1, figsize=[12,10])
		for comp in range(self.n_components):
			# first plot hidden inv-density
			h_norm = np.linalg.norm(hidden_states[0,comp,:,:], ord=2, axis=1) / np.sqrt(n_hidden_states)
			sns.kdeplot(h_norm, ax=ax_h, label='component-{comp}'.format(comp=comp))
		ax_h.set_xlabel('||h|| / sqrt(hidden-dimension)')
		ax_h.set_title('Invariant Density of Hidden State Norm')
		fig_h.savefig(fname=os.path.join(hidden_inv_dir,'epoch{epoch}'.format(epoch=epoch)))
		plt.close(fig_h)
		return


	def make_traj_plots(self, Xtrue, Xpred, Xpred_residuals, hidden_states, name, epoch):
		hidden_inv_dir = os.path.join(self.output_path, 'inv_hidden_{name}'.format(name=name))
		traj_dir = os.path.join(self.output_path, 'traj_state_{name}'.format(name=name))
		hidden_dir = os.path.join(self.output_path, 'traj_hidden_{name}'.format(name=name))
		phase_dir = os.path.join(self.output_path, 'traj_phase_{name}'.format(name=name))
		os.makedirs(phase_dir, exist_ok=True)
		os.makedirs(traj_dir, exist_ok=True)
		os.makedirs(hidden_dir, exist_ok=True)
		os.makedirs(hidden_inv_dir, exist_ok=True)

		n_hidden_states = hidden_states.shape[-1]
		n_traj, n_steps, n_states = Xtrue.shape
		n_plt = min(self.max_plot, n_steps)
		t_plot = np.linspace(0, n_plt*self.delta_t, n_plt)
		fig_h, ax_h = plt.subplots(1, 1, figsize=[12,10])
		for c in range(n_traj):
			fig, ax_list = plt.subplots(n_states, 1, figsize=[12,10], sharex=True)
			for s in range(n_states):
				ax = ax_list[s]
				ax.plot(t_plot, Xtrue[c,:n_plt,s].cpu().data.numpy(),linestyle='-', label='true')
				ax.plot(t_plot, Xpred[c,:n_plt,s].cpu().data.numpy(),linestyle='--', label='learned')
			ax.legend()
			ax.set_xlabel('Time')
			fig.suptitle('Trajectory Fit')
			fig.savefig(fname=os.path.join(traj_dir,'traj{c}_epoch{epoch}'.format(c=c,epoch=epoch)))
			plt.close(fig)

			output_fname = os.path.join(phase_dir,'traj{c}_epoch{epoch}_'.format(c=c,epoch=epoch))
			phase_plot(data=Xpred[c,:,:].cpu().data.numpy(), output_fname=output_fname, delta_t=self.delta_t, wspace=0.35, hspace=0.35)

			for comp in range(self.n_components):
				# first plot hidden inv-density
				h_norm = np.linalg.norm(hidden_states[c,comp,:,:].cpu().data.numpy(), ord=2, axis=1) / np.sqrt(n_hidden_states)
				sns.kdeplot(h_norm, ax=ax_h, label='traj-{c} component-{comp}'.format(c=c, comp=comp))

				# now plot hidden dynamics
				fig, ax = plt.subplots(1, 1, figsize=[12,10], sharex=True)
				ax.plot(t_plot, hidden_states[c,comp,:n_plt,:].cpu().data.numpy())
				ax.set_xlabel('Time')
				fig.suptitle('Hidden state dynamics')
				fig.savefig(fname=os.path.join(hidden_dir,'traj{c}_component{comp}_epoch{epoch}'.format(c=c, comp=comp, epoch=epoch)))
				plt.close(fig)

		ax_h.set_xlabel('||h|| / sqrt(hidden-dimension)')
		ax_h.set_title('Invariant Density of Hidden State Norm')
		fig_h.savefig(fname=os.path.join(hidden_inv_dir,'epoch{epoch}'.format(epoch=epoch)))
		plt.close(fig_h)

		return

	def remember_weights(self):
		for name, val in self.named_parameters(): #self.state_dict():
			norm_val = np.linalg.norm(val.cpu().data.numpy())
			if name not in self.weight_history:
				self.weight_history[name] = norm_val
			else:
				self.weight_history[name] = np.hstack((self.weight_history[name],norm_val))

	def plot_cumulative_time(self):
		self.cum_time += [(time() - self.t0_overall)/60]
		fig, ax = plt.subplots(1, 1, figsize=[8,6])
		ax.plot(self.cum_time)
		ax.set_xlabel('Epochs')
		ax.set_ylabel('Time (minutes)')
		ax.set_title('Cumulative Run Time')
		fig.savefig(fname=os.path.join(self.output_path,'rnn_run_time.png'), dpi=300)
		plt.close(fig)


	def print_epoch_status(self, model_stats, epoch=-1):
		vals = {}
		vals['epoch'] = epoch
		vals['ltrain'] = my_nanmean(model_stats['Train']['loss'][epoch,:])
		vals['ttrain'] = my_nanmedian(model_stats['Train']['t_valid'][epoch,:])
		vals['ttest'] = my_nanmedian(model_stats['Test']['t_valid'][epoch,:])
		vals['ltest'] = my_nanmean(model_stats['Test']['loss'][epoch,:])
		vals['time'] = round((time() - self.t0_epoch)/60,3)
		status_string = 'Epoch {epoch}. l-train={ltrain}, l-test={ltest}, t-train={ttrain}, t-test={ttest}, Elapsed time={time} minutes'.format(**vals)
		print(status_string)
		self.t0_epoch = time()

	def plot_weights(self, n_epochs):
		n_params = len(self.weight_history.keys())

		# plot param convergence
		fig, (axrow0, axrow1) = plt.subplots(2, 3, sharex=True, figsize=[8,6])
		axlist = np.concatenate((axrow0,axrow1))
		c = -1
		for key in self.weight_history:
			c += 1
			param_history = self.weight_history[key]
			x_vals = np.linspace(0, n_epochs, param_history.shape[0])
			if param_history.ndim==1:
				param_vals = param_history
			elif param_history.ndim==3:
				param_vals = np.linalg.norm(param_history, axis=(1,2))
			elif param_history.ndim==2:
				param_vals = np.linalg.norm(param_history, axis=(1))
			axlist[c].plot(x_vals, param_vals)
			axlist[c].set_title(key, pad=15)
			axlist[c].set_xlabel('Epochs')
		fig.suptitle("Parameter convergence")
		fig.subplots_adjust(wspace=0.3, hspace=0.3)
		fig.savefig(fname=os.path.join(self.output_path,'rnn_parameter_convergence_new.png'), dpi=300)
		plt.close(fig)

		# plot matrix visualizations
		param_path = os.path.join(self.output_path, 'params')
		os.makedirs(param_path, exist_ok=True)
		fig, (axrow0, axrow1) = plt.subplots(2, 3, sharex=False, figsize=[8,6])
		axlist = np.concatenate((axrow0,axrow1))
		c = -1
		for name, val in self.named_parameters():
			c += 1
			ax = axlist[c]
			val = val.cpu().detach()
			if val.ndim==1:
				val = val[None,:]
			if val.ndim==3:
				val = val.squeeze(0)
			foo = ax.matshow(val, vmin=torch.min(val), vmax=torch.max(val), aspect='auto')
			ax.axes.xaxis.set_visible(False)
			ax.axes.yaxis.set_visible(False)
			ax.set_title(name, pad=20)
			fig.colorbar(foo, ax=ax)

		fig.subplots_adjust(wspace=0.3, hspace=0.5)
		fig.savefig(fname=os.path.join(param_path,'rnn_parameter_values_{n_epochs}.png'.format(n_epochs=n_epochs-1)), dpi=300)
		plt.close(fig)


def get_optimizer(params, name='SGD', lr=None):
	if name=='SGD':
		if lr is None:
			lr = 0.05
		return optim.SGD(params, lr=lr)
	elif name=='Adam':
		# if lr is None:
		# 	lr = lr=0.001
		# return optim.Adam(params, lr=lr)
		print('Ignoring learning rate and using Adam defaults')
		return optim.Adam(params)
	elif name=='LBFGS':
		if lr is None:
			lr = 1
		return optim.LBFGS(params, lr=lr)
	elif name=='RMSprop':
		if lr is None:
			lr = 0.01
		return optim.RMSprop(params, lr=lr)
	else:
		return None

def get_loss(name='nn.MSELoss', weight=None):
	if name == 'nn.MSELoss':
		return nn.MSELoss()
	else:
		raise('Loss name not recognized')

def train_RNN_new(
				y_clean_test_long,
				y_noisy_train,
				y_noisy_test,
				y_noisy_testSynch,
				model_params=None,
				output_dir='.',
				n_grad_steps=1,
				num_frames=None,
				n_epochs=10,
				save_freq=None,
				use_physics_as_bias=False,
				use_gpu=True,
				normz_info=None,
				ODE=None,
				mode=None,
				do_printing=False,
				component_wise=False,
				cell_type='RNN',
				hidden_size=50,
				use_manual_seed=False,
				old_optim=False,
				optimizer_name='SGD',
				lr=0.05,
				early_save_fraction=0.66,
				**kwargs):

	if not save_freq:
		save_freq = max(int(n_epochs/10),1)

	n_test_traj = y_noisy_test.shape[0]
	n_train_traj = 1 #y_noisy_train.shape[0]

	model_stats = {'Train': {'loss': np.zeros((n_epochs,n_train_traj))*np.nan,
							't_valid': np.zeros((n_epochs,n_train_traj))*np.nan
							},
					'Test': {'loss': np.zeros((n_epochs,n_test_traj))*np.nan,
							't_valid': np.zeros((n_epochs,n_test_traj))*np.nan
							}
						}

	output_path = output_dir
	os.makedirs(output_path, exist_ok=True)

	if use_gpu and not torch.cuda.is_available():
		# https://thedavidnguyenblog.xyz/installing-pytorch-1-0-stable-with-cuda-10-0-on-windows-10-using-anaconda/
		print('Trying to use GPU, but cuda is NOT AVAILABLE. Running with CPU instead.')
		use_gpu = False

	# choose cuda-GPU or regular
	if use_gpu:
		dtype = torch.cuda.FloatTensor
		inttype = torch.cuda.IntTensor
	else:
		dtype = torch.float
		inttype = torch.int

	# get data
	Xtrain = y_noisy_train[None,:-1,:]
	ytrain = y_noisy_train[None,1:,:]
	Xtest = y_noisy_test
	Xtest_long = y_clean_test_long

	# ytest = y_noisy_test[:,1:,:]
	Xtest_synch = y_noisy_testSynch[:,:-1,:]
	ytest_synch = y_noisy_testSynch[:,1:,:]
	Xtest_init = y_noisy_testSynch[:,None,-1,:]

	# get unnormalized data
	Xtrain_raw = unnormalize(norm_dict=normz_info, y_norm=Xtrain)
	ytrain_raw = unnormalize(norm_dict=normz_info, y_norm=ytrain)
	Xtest_raw = unnormalize(norm_dict=normz_info, y_norm=Xtest)
	# ytest_raw = unnormalize(norm_dict=normz_info, y_norm=ytest)

	Xtest_synch_raw = unnormalize(norm_dict=normz_info, y_norm=Xtest_synch)
	ytest_synch_raw = unnormalize(norm_dict=normz_info, y_norm=ytest_synch)
	Xtest_long_raw = unnormalize(norm_dict=normz_info, y_norm=Xtest_long)

	# set up model
	model = RNN(ode_params=model_params,
				input_size=Xtrain.shape[2],
				norm_dict=normz_info,
				use_physics_as_bias=use_physics_as_bias,
				ode=ODE,
				output_path=output_path,
				mode=mode,
				component_wise=component_wise,
				cell_type=cell_type,
				hidden_size=hidden_size,
				use_manual_seed=use_manual_seed,
				dtype=dtype)
	model.remember_weights()

	# generate bias sequences
	if use_physics_as_bias:
		model.solver_failed = [False]
		Xtrain_pred = model.normalize(model.get_physics_prediction(X=model.unnormalize(torch.FloatTensor(Xtrain).type(dtype))))

	if use_gpu:
		model.cuda()
	optimizer = get_optimizer(name=optimizer_name, params=model.parameters(), lr=lr)
	loss_function = get_loss()

	best_model_dict = {}

	# train the model
	train_loss_vec = np.zeros((n_epochs,1))
	test_loss_vec = np.zeros((n_epochs,1))

	if num_frames is None:
		num_frames = Xtrain.shape[1]

	best_test_loss = np.inf
	best_test_tvalid = 0
	for epoch in range(n_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
		t0 = time()
		# setence is our features, tags are INDICES of true label
		all_predicted_states = []
		all_target_states = []
		all_rnn_predicted_residuals = []
		all_hidden_states = []
		if num_frames == Xtrain.shape[1]:
			# 1 big chunk of the whole data set
			offset = 0
			chunk_list = [0]
		elif num_frames == 1:
			# don't reorder...just update gradient at each Step
			offset = 0
			chunk_list = np.arange(Xtrain.shape[1]-1)
		else:
			# chunk strategy
			offset = np.random.randint(Xtrain.shape[1] % num_frames)
			chunk_list = np.arange(int(Xtrain.shape[1] / num_frames)).tolist()
			np.random.shuffle(chunk_list)

		for chunk_ind in chunk_list:
			# identify a random chunk of data
			start_chunk = chunk_ind * num_frames + offset
			end_chunk = start_chunk + num_frames

			i_stop = start_chunk
			model.clear_hidden() # set hidden state to 0
			model.zero_grad() # reset gradients dL/dparam
			model.remember_weights() # store history of parameter updates
			while i_stop < end_chunk:
				i_start = i_stop
				i_stop = min(i_stop+n_grad_steps, end_chunk)

				# get bias sequence
				# print('This bias sequence should be coming from Psi_epsilon, not the real data! Maybe think about doing teacher forcing here.')
				if use_physics_as_bias:
					bias_sequence = Xtrain_pred[:,i_start:i_stop,:] #normalized
				else:
					bias_sequence = None

				# Run our forward pass. with normalized inputs
				full_predicted_states, rnn_predicted_residuals, hidden_states = model(input_state_sequence=torch.FloatTensor(Xtrain[:,i_start:i_stop,:]).type(dtype),
												physical_prediction_sequence=bias_sequence, train=True)

				# fit the RNN to normalized outputs
				target_sequence = torch.FloatTensor(ytrain[:,i_start:i_stop,:]).type(dtype)


				### OLD style of updates
				# loss = loss_function(full_predicted_states, target_sequence) #compute loss
				loss = (full_predicted_states.squeeze() - target_sequence.squeeze()).pow(2).sum()
				loss.backward()

				if do_printing:
					for name, val in model.named_parameters():
						if val.requires_grad:
							easy_name = model.lookup[name][0]
							print('|{0}|'.format(easy_name), np.linalg.norm(val.data))
					for name, val in model.named_parameters():
						if val.requires_grad:
							easy_name = model.lookup[name][0]
							print('|grad_{0}|'.format(easy_name), np.linalg.norm(val.grad.data))

				# https://discuss.pytorch.org/t/losses-not-matching-when-training-using-hand-written-sgd-and-torch-optim-sgd/3467/3
				if old_optim:
					for name, val in model.named_parameters():
						if val.requires_grad:
							val.data -= lr* val.grad.data / n_grad_steps # use avg gradient per step
							# val.data.add_(val.grad, alpha=-lr)
							val.grad.data.zero_()
				else:
					optimizer.step() # update parameters using dL/dparam
					model.zero_grad() # reset gradients dL/dparam

				if do_printing:
					for name, val in model.named_parameters():
						if val.requires_grad:
							easy_name = model.lookup[name][0]
							# print('|{0}|'.format(easy_name), np.linalg.norm(val.data))
					print('hidden state:', model.h_t)
					print('target:', target_sequence)
					print('pred:', full_predicted_states)
					print('loss:', loss)
					# pdb.set_trace()

				model.detach_hidden() # remove hidden-states from graph so that gradients at next step are not dependent on previous step
				model.remember_weights() # store history of parameter updates
				### end OLD style of updates

				### NEW style of updates
				# Compute the loss, gradients, and update the parameters by
				#  calling optimizer.step()
				# loss = 3*loss_function(full_predicted_states, target_sequence) #compute loss
				# old_loss = (full_predicted_states.squeeze() - target_sequence.squeeze()).pow(2).sum()
				# if not np.isclose(loss.detach().numpy(), old_loss.detach().numpy()):
				# 	pdb.set_trace()
				# loss.backward() # compute dL/dparam for each param via BPTT
				# optimizer.step() # update parameters using dL/dparam
				# model.zero_grad() # reset gradients dL/dparam
				# model.h_t.detach_() # remove hidden-state from graph so that gradients at next step are not dependent on previous step
				# model.remember_weights() # store history of parameter updates
				### end NEW style of updates

				# collect the data
				all_predicted_states.append(full_predicted_states)
				all_target_states.append(target_sequence)
				all_rnn_predicted_residuals.append(rnn_predicted_residuals)
				all_hidden_states.append(hidden_states)

		# collect and nnormalize the data
		all_target_states = model.unnormalize(torch.cat(all_target_states, 1).cpu())
		all_predicted_states = model.unnormalize(torch.cat(all_predicted_states, 1).cpu())
		all_rnn_predicted_residuals = model.unnormalize(torch.cat(all_rnn_predicted_residuals, 1).cpu())
		all_hidden_states = torch.cat(all_hidden_states, 2).cpu() # no need to unnormalize hidden states

		# Report Train losses after each epoch
		for c in range(n_train_traj):
			model_stats['Train']['t_valid'][epoch,c] = traj_div_time(Xtrue=all_target_states[c,:,:], Xpred=all_predicted_states[c,:,:], delta_t=model.delta_t, avg_output=model.time_avg_norm, synch_length=Xtest_synch.shape[1])
			model_stats['Train']['loss'][epoch,c] = all_target_states.shape[2]*loss_function(model.normalize(all_target_states[c,:,:]), model.normalize(all_predicted_states[c,:,:])).cpu().data.numpy().item()

		if (epoch / n_epochs) < early_save_fraction:
			# Print epoch summary after every epoch
			model.print_epoch_status(model_stats, epoch)
			model.plot_cumulative_time()

			# Plot intermittent stuff after 10% increments
			is_save_interval = (epoch % save_freq == 0)
			if is_save_interval:
				plot_stats(model_stats, epoch=epoch+1, output_path=output_path)
				model.plot_weights(n_epochs=epoch+1)
				model.make_traj_plots(all_target_states, all_predicted_states, all_rnn_predicted_residuals, all_hidden_states, name='train', epoch=epoch)
			continue

		### Report TEST performance after each epoch
		# Step 0. reset initial hidden states
		model.clear_hidden()

		# Step 1. Run forward pass with normalized synchronization data
		full_predicted_states_synch, rnn_predicted_residuals_synch, hidden_states_synch = model(input_state_sequence=torch.FloatTensor(Xtest_synch).type(dtype),
										physical_prediction_sequence=None, train=False, synch_mode=True)

		# Step 2. Run our forward pass with synchronized RNN
		# Note that bias-terms and physical predictions must be computed on the fly
		full_predicted_states_test, rnn_predicted_residuals_test, hidden_states_test = model(input_state_sequence=torch.FloatTensor(Xtest_init).type(dtype),
										n_steps = Xtest.shape[1],
										physical_prediction_sequence=None, train=False, synch_mode=False)

		# Step 3. Compute the test-traj losses
		test_loss = (full_predicted_states_test.squeeze() - torch.FloatTensor(Xtest).type(dtype).squeeze()).pow(2).sum()
		# test_loss = loss_function(full_predicted_states_test, torch.FloatTensor(ytest).type(dtype)).detach().numpy()
		# print('Test Loss:', test_loss)

		# unnormalize the test outputs
		target_sequence_test = torch.FloatTensor(Xtest_raw).type(dtype)
		full_predicted_states_test = model.unnormalize(full_predicted_states_test.cpu())
		rnn_predicted_residuals_test = model.unnormalize(rnn_predicted_residuals_test.cpu())
		# unnormalize the test_synch outputs
		target_sequence_synch = torch.FloatTensor(ytest_synch_raw).type(dtype)
		full_predicted_states_synch = model.unnormalize(full_predicted_states_synch.cpu())
		rnn_predicted_residuals_synch = model.unnormalize(rnn_predicted_residuals_synch.cpu())


		for c in range(n_test_traj):
			model_stats['Test']['t_valid'][epoch,c] = traj_div_time(Xtrue=target_sequence_test[c,:,:], Xpred=full_predicted_states_test[c,:,:], delta_t=model.delta_t, avg_output=model.time_avg_norm)
			model_stats['Test']['loss'][epoch,c] = target_sequence_test.shape[2]*loss_function(model.normalize(target_sequence_test[c,:,:]), model.normalize(full_predicted_states_test[c,:,:])).cpu().data.numpy().item()
		test_tvalid = np.mean(model_stats['Test']['t_valid'][epoch,:])

		# Print epoch summary after every epoch
		model.print_epoch_status(model_stats, epoch)
		model.plot_cumulative_time()

		# Plot intermittent stuff after 10% increments
		has_improved_loss = test_loss < best_test_loss
		has_improved_tvalid =  test_tvalid > best_test_tvalid
		is_save_interval = (epoch % save_freq == 0)
		if has_improved_loss:
			best_test_loss = test_loss
		if has_improved_tvalid:
			best_test_tvalid = test_tvalid
		if has_improved_loss or has_improved_tvalid or is_save_interval or (epoch==n_epochs-1):
			plot_stats(model_stats, epoch=epoch+1, output_path=output_path)
			model.plot_weights(n_epochs=epoch+1)
			model.make_traj_plots(all_target_states, all_predicted_states, all_rnn_predicted_residuals, all_hidden_states, name='train', epoch=epoch)
			model.make_traj_plots(target_sequence_test, full_predicted_states_test, rnn_predicted_residuals_test, hidden_states_test, name='test', epoch=epoch)
			model.make_traj_plots(target_sequence_synch, full_predicted_states_synch, rnn_predicted_residuals_synch, hidden_states_synch, name='test_synch', epoch=epoch)

			#### Step 4. Now, test the invariant measure (use only 1 synch traj)
			# NOTE: we synchronize using the test-traj synchronization data, not a huge burn-in
			model.h_t = model.h_t[:,None,0,:] #model.h_t[None,:,0,:]
			if model.use_c_cell:
				model.c_t = model.c_t[:,None,0,:]
			model(input_state_sequence=torch.FloatTensor(Xtest_synch[None,0,:]).type(dtype), physical_prediction_sequence=None, train=False, synch_mode=True)
			full_predicted_states_test_long, rnn_predicted_residuals_test_long, hidden_states_test_long = model(input_state_sequence=torch.FloatTensor(Xtest_init[None,0]).type(dtype),
											n_steps = Xtest_long.shape[0],
											physical_prediction_sequence=None, train=False, synch_mode=False)
			model.make_invariant_measure_plots(Xtrue=Xtest_long_raw, Xpred=model.unnormalize(full_predicted_states_test_long.squeeze()).cpu().data.numpy(), hidden_states=hidden_states_test_long.cpu().data.numpy(), epoch=epoch)


		if is_save_interval:
			for name, val in model.named_parameters():
				if val.requires_grad:
					try:
						easy_name = model.lookup[name][0]
					except:
						easy_name = name
					print('|{0}|'.format(easy_name), np.linalg.norm(val.data))

	print('all done!')

def my_nanmean(x, axis=None):
	with warnings.catch_warnings():
		warnings.filterwarnings('error')
		try:
			val = np.nanmean(x, axis=axis)
		except RuntimeWarning:
			val = np.mean(x, axis=axis)
	return val

def my_nanmedian(x, axis=None):
	with warnings.catch_warnings():
		warnings.filterwarnings('error')
		try:
			val = np.nanmedian(x, axis=axis)
		except RuntimeWarning:
			val = np.median(x, axis=axis)
	return val

def my_nanstd(x, axis=None):
	with warnings.catch_warnings():
		warnings.filterwarnings('error')
		try:
			val = np.nanstd(x, axis=axis)
		except RuntimeWarning:
			val = np.std(x, axis=axis)
	return val

def plot_stats(model_stats, epoch=-1, output_path='.'):
	train_loss_vec = model_stats['Train']['loss']
	train_t_valid_vec = model_stats['Train']['t_valid']
	test_loss_vec = model_stats['Test']['loss']
	test_t_valid_vec = model_stats['Test']['t_valid']

	fig, ax_list = plt.subplots(2,2, figsize=[12,10], sharex=True)

	# loss function
	ax = ax_list[0,0]
	ax.errorbar(x=np.arange(epoch), y=my_nanmean(train_loss_vec[:epoch,:], axis=1), yerr=my_nanstd(train_loss_vec[:epoch,:], axis=1), label='Training Loss', linestyle='-')
	ax.set_title('Training Error')
	ax.set_ylabel('Loss')

	ax = ax_list[0,1]
	ax.errorbar(x=np.arange(epoch), y=my_nanmean(test_loss_vec[:epoch,:], axis=1), yerr=my_nanstd(test_loss_vec[:epoch,:], axis=1), label='Testing Loss', linestyle='--')
	ax.set_title('Testing Error')
	ax.set_ylabel('Loss')

	# validity time
	ax = ax_list[1,0]
	ax.errorbar(x=np.arange(epoch), y=my_nanmean(train_t_valid_vec[:epoch,:], axis=1), yerr=my_nanstd(train_t_valid_vec[:epoch,:], axis=1), label=' Train', linestyle='-')
	ax.set_title('Training Validity Time')
	ax.set_ylabel('Validity Time')

	ax = ax_list[1,1]
	ax.errorbar(x=np.arange(epoch), y=my_nanmean(test_t_valid_vec[:epoch,:], axis=1), yerr=my_nanstd(test_t_valid_vec[:epoch,:], axis=1), label=' Test', linestyle='--')
	ax.set_title('Testing Validity Time')
	ax.set_ylabel('Validity Time')

	fig.savefig(fname=os.path.join(output_path,'TrainTest_Performance'))
	plt.close(fig)

	## Plot Train vs Test Correlations
	fig, ax_list = plt.subplots(1,2, figsize=[16,8], sharex=True)

	train_loss = my_nanmean(train_loss_vec[:epoch,:], axis=1)
	test_loss = my_nanmean(test_loss_vec[:epoch,:], axis=1)
	test_t_valid = my_nanmean(test_t_valid_vec[:epoch,:], axis=1)

	# loss function
	ax = ax_list[0]
	ax.plot(train_loss, test_loss, 'o', markersize=4)
	ax.set_xlabel('Train Loss')
	ax.set_ylabel('Test Loss')

	ax = ax_list[1]
	ax.plot(train_loss, test_t_valid, 'o', markersize=4)
	ax.set_xlabel('Train Loss')
	ax.set_ylabel('Test Validity Time')

	fig.savefig(fname=os.path.join(output_path,'TrainTest_Correlations'))

	ax_list[0].set_xscale('log')
	ax_list[0].set_xscale('log')
	ax_list[1].set_xscale('log')
	fig.savefig(fname=os.path.join(output_path,'TrainTest_Correlations_log'))

	plt.close(fig)
	return
