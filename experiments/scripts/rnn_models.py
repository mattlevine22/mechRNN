import os, sys
import numpy as np
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from utils import traj_div_time, train_chaosRNN
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

	# read TRAIN data
	train_set = np.load(training_fname)
	y_clean_train = train_set['X_train']
	y_noisy_train = train_set['X_train']

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
	y_clean_test = []
	y_noisy_test = []
	y_clean_testSynch = []
	y_noisy_testSynch = []
	for c in range(test_set['X_test_traj'].shape[0]):
		y_clean_test.append(normalize(norm_dict=normz_info, y=test_set['X_test_traj'][c,:,:]))
		y_noisy_test.append(normalize(norm_dict=normz_info, y=test_set['X_test_traj'][c,:,:]))
		y_clean_testSynch.append(normalize(norm_dict=normz_info, y=test_set['X_test_traj_synch'][c,:,:]))
		y_noisy_testSynch.append(normalize(norm_dict=normz_info, y=test_set['X_test_traj_synch'][c,:,:]))

	setts['y_clean_test'] = np.stack(y_clean_test)
	setts['y_noisy_test'] = np.stack(y_noisy_test)
	setts['y_clean_testSynch'] = np.stack(y_clean_testSynch)
	setts['y_noisy_testSynch'] = np.stack(y_noisy_testSynch)

	# get state names
	setts['model_params']['state_names'] = odeInst.get_state_names()

	setts['model'] = odeInst.rhs

	setts['plot_state_indices'] = odeInst.plot_state_indices()

	setts['ODE'] = odeInst


	if profile:
		lp = LineProfiler()
		# lp.add_function(forward_chaos_pureML)
		# lp.add_function(forward_chaos_hybrid_full)
		lp_wrapper = lp(train_RNN_new)
		lp_wrapper(**setts)
		lp.print_stats()
	else:
		setts['output_dir'] += '_old'
		train_chaosRNN(**setts)
		setts['output_dir'] = setts['output_dir'].replace('old','new')
		setts['mode'] = 'original'
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
			mode='original'):

		super().__init__()
		if output_size is None:
			output_size = input_size
		self.mode = mode
		self.output_path = output_path
		self.teacher_force_probability = teacher_force_probability
		self.t_synch = t_synch
		self.dtype = dtype
		self.hidden_size = hidden_size
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
			self.input_size = 2*input_size
		else:
			self.input_size = input_size

		#This is how to add a parameter
		# self.w = nn.Parameter(scalar(0.1), requires_grad=True)

		# Default is RNN w/ ReLU
		if cell_type=='LSTM':
			self.cell = nn.LSTMCell(input_size, hidden_size)
			self.use_c_cell = True
		elif cell_type=='GRU':
			self.cell = nn.GRUCell(input_size, hidden_size)
			self.use_c_cell = True
		else:
			self.use_c_cell = False
			self.cell = nn.RNNCell(input_size, hidden_size, nonlinearity='relu')

		self.lookup = {'cell.weight_hh': 'A_mat',
					'cell.weight_ih': 'B_mat',
					'hidden2pred.weight': 'C_mat',
					'cell.bias_hh': 'a_vec',
					'hidden2pred.bias': 'b_vec'}


		# The linear layer that maps from hidden state space to tag space
		self.hidden2pred = nn.Linear(hidden_size, output_size)

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
				nn.init.normal_(val, mean=0.0, std=0.1)
		elif self.mode=='fromfile':
			for name, val in self.named_parameters():
				# pdb.set_trace()
				if name in self.lookup:
					fname = os.path.join('/Users/matthewlevine/test_outputs/l63/rnn_output/10_epochs/pureRNN_vanilla_old', self.lookup[name] + '.txt')
					val.data = torch.FloatTensor(np.loadtxt(fname=fname)).type(self.dtype)


		# CUSTOM slight adjustment to parameterization
		# for some reason, pytorch includes 2 redundant bias terms in the cell
		self.cell.bias_ih.data = torch.zeros(self.cell.bias_ih.data.shape)
		self.cell.bias_ih.requires_grad = False
		return

	def clear_hidden(self):
		self.h_t = None
		self.c_t = None

	def make_traj_plots(self, Xtrue, Xpred, Xpred_residuals, hidden_states, name, epoch):
		traj_dir = os.path.join(self.output_path, 'traj_state_{name}'.format(name=name))
		hidden_dir = os.path.join(self.output_path, 'traj_hidden_{name}'.format(name=name))
		os.makedirs(traj_dir, exist_ok=True)
		os.makedirs(hidden_dir, exist_ok=True)

		n_traj, n_steps, n_states = Xtrue.shape
		n_plt = min(self.max_plot, n_steps)
		t_plot = np.linspace(0,self.delta_t, n_plt)
		for c in range(n_traj):
			fig, ax_list = plt.subplots(n_states, 1, figsize=[12,10], sharex=True)
			for s in range(n_states):
				ax = ax_list[s]
				ax.plot(t_plot, Xtrue[c,:n_plt,s].cpu().data.numpy(),linestyle='-', label='true')
				ax.plot(t_plot, Xpred[c,:n_plt,s].cpu().data.numpy(),linestyle='--', label='learned')
			ax.legend()
			ax.set_xlabel('Time')
			fig.suptitle('Trajectory Fit')
			fig.savefig(fname=os.path.join(traj_dir,'trajfit{c}_epoch{epoch}'.format(c=c,epoch=epoch)))
			plt.close(fig)

			fig, ax = plt.subplots(1, 1, figsize=[12,10], sharex=True)
			ax.plot(t_plot, hidden_states[c,:n_plt,:].cpu().data.numpy())
			ax.set_xlabel('Time')
			fig.suptitle('Hidden state dynamics')
			fig.savefig(fname=os.path.join(hidden_dir,'hiddenstate{c}_epoch{epoch}'.format(c=c,epoch=epoch)))
			plt.close(fig)
		return

	def remember_weights(self):
		for name, val in self.named_parameters(): #self.state_dict():
			norm_val = np.linalg.norm(val.data.numpy())
			if name not in self.weight_history:
				self.weight_history[name] = norm_val
			else:
				self.weight_history[name] = np.hstack((self.weight_history[name],norm_val))
		return

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
		fig.savefig(fname=os.path.join(self.output_path,'rnn_parameter_convergence.png'), dpi=300)
		plt.close(fig)

		# plot matrix visualizations
		param_path = os.path.join(self.output_path, 'params')
		os.makedirs(param_path, exist_ok=True)
		fig, (axrow0, axrow1) = plt.subplots(2, 3, sharex=True, figsize=[8,6])
		axlist = np.concatenate((axrow0,axrow1))
		c = -1
		for name, val in self.named_parameters():
			ax = axlist[c]
			val = val.detach()
			c += 1
			if val.ndim==1:
				val = val[None,:]
			if val.ndim==3:
				val = val.squeeze(0)
			# pdb.set_trace()
			foo = ax.matshow(val, vmin=torch.min(val), vmax=torch.max(val))
			ax.axes.xaxis.set_visible(False)
			ax.axes.yaxis.set_visible(False)
			ax.set_title(name, pad=20)
			fig.colorbar(foo, ax=ax)

		# fig.suptitle("Parameter convergence")
		fig.subplots_adjust(wspace=0.3, hspace=0.5)
		fig.savefig(fname=os.path.join(param_path,'rnn_parameter_values_{n_epochs}.png'.format(n_epochs=n_epochs-1)), dpi=300)
		plt.close(fig)


		return

	def normalize(self, y):
		return normalize(norm_dict=self.norm_dict, y=y)

	def unnormalize(self, y_norm):
		return unnormalize(norm_dict=self.norm_dict, y_norm=y_norm)

	def get_physics_prediction(self, ic):
		#input and output are unnormalized
		n_ics = ic.shape[0]

		y_pred = np.zeros(ic.shape)
		for c in range(n_ics):
			# check if bad initial condition
			if (any(abs(ic[c,:])>1000) or any(np.isnan(ic[c,:]))):
				if not self.solver_failed[c]: # only print if it is new/recent news!
					# pdb.set_trace()
					print('ODE initial conditions are huge, so not even trying to solve the system. Applying the Identity forward map instead.',ic[c,:])
				self.solver_failed[c] = True
			else:
				self.solver_failed[c] = False

			if not self.solver_failed[c]:
				sol = solve_ivp(fun=lambda t, y: self.ode.rhs(y, t), t_span=self.tspan, y0=ic[c,:], t_eval=self.t_eval, **self.ode_params)
				# sol = solve_ivp(fun=lambda t, y: model(y, t, *model_params['ode_params']), t_span=(self.tspan[0], self.tspan[-1]), y0=y0.T, method=model_params['ode_int_method'], rtol=model_params['ode_int_rtol'], atol=model_params['ode_int_atol'], max_step=model_params['ode_int_max_step'], t_eval=self.tspan)
				y_out = sol.y.T
				if not sol.success:
					# solver failed
					print('ODE solver has failed at ic=',ic[c,:])
					self.solver_failed[c] = True

			if self.solver_failed[c]:
				y_pred[c,:] = np.copy(ic[c,:]) # persist previous solution
			else:
				# solver is OKAY--use the solution like a good boy!
				y_pred[c,:] = y_out[-1,:]

		return torch.FloatTensor(y_pred).type(self.dtype)

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
			self.h_t = torch.zeros((input_state_sequence.size(0), self.hidden_size), dtype=self.dtype, requires_grad=True) # (batch, hidden_size)
			if train:
				nn.init.normal_(self.h_t,0.0,0.1)
		if self.c_t is None and self.use_c_cell:
			self.c_t = torch.zeros((input_state_sequence.size(0), self.hidden_size), dtype=self.dtype, requires_grad=True) # (batch, hidden_size)
		full_rnn_pred = input_state_sequence[:,0] #normalized

		self.solver_failed = [False for _ in range(full_rnn_pred.shape[0])]
		# consider Scheduled Sampling (https://arxiv.org/abs/1506.03099) where probability of using RNN-output increases as you train.
		for t in range(n_steps):
			# get input to hidden state
			if train:
				# consider teacher forcing (using RNN output prediction as next training input instead of training data)
				if t>self.t_synch and random.random()<self.teacher_force_probability:
					input_t = full_rnn_pred #feed RNN prediction back in as next input
				else:
					input_t = input_state_sequence[:,t,:]
			else:
				if synch_mode:
					input_t = input_state_sequence[:,t,:]
				else:
					input_t = full_rnn_pred

			if self.use_physics_as_bias or self.embed_physics_prediction:
				if physical_prediction_sequence is not None:
					physics_pred = physical_prediction_sequence[:,t,:]
				else:
					if input_t.ndim==1:
						input_t = input_t[None,:]
					ic = self.unnormalize(input_t).detach().numpy()
					physics_pred = self.normalize(self.get_physics_prediction(ic=ic))

				if self.embed_physics_prediction:
					input_t = torch.stack(input_t, physics_pred)
			else:
				physics_pred = 0

			# evolve hidden state
			if self.use_c_cell: # LSTM / GRU
				self.h_t, self.c_t = self.cell(input_t, (self.h_t, self.c_t)) # input needs to be dim (batch, input_size)
			else: # standard RNN
				self.h_t = self.cell(input_t, self.h_t) # input needs to be dim (batch, input_size)

			rnn_pred = self.hidden2pred(self.h_t)
			full_rnn_pred = self.use_physics_as_bias * physics_pred + rnn_pred # normalized
			full_preds += [full_rnn_pred]
			rnn_preds += [rnn_pred]
			hidden_preds += [self.h_t]

		full_preds = torch.stack(full_preds, 1).squeeze(2)
		rnn_preds = torch.stack(rnn_preds, 1).squeeze(2)
		hidden_preds = torch.stack(hidden_preds, 1).squeeze(2)

		return full_preds, rnn_preds, hidden_preds


def get_optimizer(params, name='SGD', lr=None):
	if name=='SGD':
		if lr is None:
			lr = 0.0005
		return optim.SGD(params, lr=lr)
	elif name=='Adam':
		if lr is None:
			lr = 0.01
		return optim.Adam(params, lr=lr)
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

def train_RNN_new(y_noisy_train,
				y_noisy_test,
				y_noisy_testSynch,
				model_params=None,
				output_dir='.',
				n_grad_steps=1,
				num_frames=None,
				n_epochs=10,
				save_freq=None,
				use_physics_as_bias=False,
				use_gpu=False,
				normz_info=None,
				ODE=None,
				mode='original',
				lr=0.05,
				do_printing=False,
				**kwargs):

	if not save_freq:
		save_freq = int(n_epochs/10)

	n_test_traj = y_noisy_test.shape[0]
	n_train_traj = 1 #y_noisy_train.shape[0]

	model_stats = {'Train': {'loss': np.zeros((n_epochs,n_train_traj)),
							't_valid': np.zeros((n_epochs,n_train_traj))
							},
					'Test': {'loss': np.zeros((n_epochs,n_test_traj)),
							't_valid': np.zeros((n_epochs,n_test_traj))
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
		dtype = torch.cuda.float
		inttype = torch.cuda.int
	else:
		dtype = torch.float
		inttype = torch.int

	# get data
	Xtrain = y_noisy_train[None,:-1,:]
	ytrain = y_noisy_train[None,1:,:]
	Xtest = y_noisy_test
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

	# set up model
	model = RNN(ode_params=model_params, input_size=Xtrain.shape[2], norm_dict=normz_info, use_physics_as_bias=use_physics_as_bias, ode=ODE, output_path=output_path, mode=mode)
	model.remember_weights()

	if use_gpu:
		model.cuda()
	optimizer = get_optimizer(params=model.parameters(), lr=lr)
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
					bias_sequence = None #torch.FloatTensor(Xtrain[:,i_start:i_stop,:]).type(dtype) #normalized
				else:
					bias_sequence = None

				# Run our forward pass. with normalized inputs
				# pdb.set_trace()
				full_predicted_states, rnn_predicted_residuals, hidden_states = model(input_state_sequence=torch.FloatTensor(Xtrain[:,i_start:i_stop,:]).type(dtype),
												physical_prediction_sequence=bias_sequence, train=True)

				# fit the RNN to normalized outputs
				target_sequence = torch.FloatTensor(ytrain[:,i_start:i_stop,:]).type(dtype)


				### OLD style of updates #####
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
				# manual_new = {}
				# manual2_new = {}
				for name, val in model.named_parameters():
					if val.requires_grad:
						easy_name = model.lookup[name][0]
						# manual_new[easy_name] = np.linalg.norm(val.data - lr*val.grad.data)
						val.data -= lr* val.grad.data
						# val.data.add_(val.grad, alpha=-lr)
						# manual2_new[easy_name] = np.linalg.norm(val.data)
						val.grad.data.zero_()
						# if manual2_new[easy_name] != manual_new[easy_name]:
						# 	pdb.set_trace()

						# if do_printing:
						# 	print('Manual New |{0}|'.format(easy_name), np.linalg.norm(val.data - lr*val.grad.data))

				# optimizer.step() # update parameters using dL/dparam
				# model.zero_grad() # reset gradients dL/dparam

				# optim_new = {}
				# for name, val in model.named_parameters():
				# 	if val.requires_grad:
				# 		easy_name = model.lookup[name][0]
				# 		optim_new[easy_name] = np.linalg.norm(val.data)
				# 		# val.data -= lr* val.grad.data
				# 		# val.grad.data.zero_()
				# 		# if do_printing:
				# 		# 	print('Optim New |{0}|'.format(easy_name), np.linalg.norm(val.data))
				# 		if optim_new[easy_name] != manual_new[easy_name]:
				# 			pdb.set_trace()

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

				# A.data -= lr * A.grad.data
				# B.data -= lr * B.grad.data
				# C.data -= lr * C.grad.data
				# a.data -= lr * a.grad.data
				# b.data -= lr * b.grad.data

				# A.grad.data.zero_()
				# B.grad.data.zero_()
				# C.grad.data.zero_()
				# a.grad.data.zero_()
				# b.grad.data.zero_()

				# hidden_state = hidden_state.detach()
				model.h_t.detach_() # remove hidden-state from graph so that gradients at next step are not dependent on previous step
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
		all_target_states = model.unnormalize(torch.cat(all_target_states, 1))
		all_predicted_states = model.unnormalize(torch.cat(all_predicted_states, 1))
		all_rnn_predicted_residuals = model.unnormalize(torch.cat(all_rnn_predicted_residuals, 1))
		all_hidden_states = torch.cat(all_hidden_states, 1) # no need to unnormalize hidden states

		# Report Train losses after each epoch
		for c in range(n_train_traj):
			model_stats['Train']['t_valid'][epoch,c] = traj_div_time(Xtrue=all_target_states[c,:,:], Xpred=all_predicted_states[c,:,:], delta_t=model.delta_t, avg_output=model.time_avg_norm, synch_length=Xtest_synch.shape[1])
			model_stats['Train']['loss'][epoch,c] = loss_function(all_target_states[c,:,:], all_predicted_states[c,:,:]).cpu().data.numpy().item()

		### Report TEST performance after each epoch
		# Step 0. reset initial hidden states
		model.clear_hidden()

		# Step 1. Run forward pass with normalized synchronization data
		full_predicted_states_synch, rnn_predicted_residuals_synch, hidden_states_synch = model(input_state_sequence=torch.FloatTensor(Xtest_synch).type(dtype),
										physical_prediction_sequence=None, train=False, synch_mode=True)
		# pdb.set_trace()

		# Step 2. Run our forward pass with synchronized RNN
		# Note that bias-terms and physical predictions must be computed on the fly
		full_predicted_states_test, rnn_predicted_residuals_test, hidden_states_test = model(input_state_sequence=torch.FloatTensor(Xtest_init).type(dtype),
										n_steps = Xtest.shape[1],
										physical_prediction_sequence=None, train=False, synch_mode=False)

		# Step 3. Compute the losses
		test_loss = (full_predicted_states_test.squeeze() - torch.FloatTensor(Xtest).type(dtype).squeeze()).pow(2).sum()
		# test_loss = loss_function(full_predicted_states_test, torch.FloatTensor(ytest).type(dtype)).detach().numpy()
		# print('Test Loss:', test_loss)
		# pdb.set_trace()

		# unnormalize the test outputs
		target_sequence_test = torch.FloatTensor(Xtest_raw).type(dtype)
		full_predicted_states_test = model.unnormalize(full_predicted_states_test)
		rnn_predicted_residuals_test = model.unnormalize(rnn_predicted_residuals_test)
		# unnormalize the test_synch outputs
		target_sequence_synch = torch.FloatTensor(ytest_synch_raw).type(dtype)
		full_predicted_states_synch = model.unnormalize(full_predicted_states_synch)
		rnn_predicted_residuals_synch = model.unnormalize(rnn_predicted_residuals_synch)


		for c in range(n_test_traj):
			model_stats['Test']['t_valid'][epoch,c] = traj_div_time(Xtrue=target_sequence_test[c,:,:], Xpred=full_predicted_states_test[c,:,:], delta_t=model.delta_t, avg_output=model.time_avg_norm)
			model_stats['Test']['loss'][epoch,c] = loss_function(target_sequence_test[c,:,:], full_predicted_states_test[c,:,:]).cpu().data.numpy().item()
		test_tvalid = np.median(model_stats['Test']['t_valid'][epoch,:])

		# Print epoch summary after every epoch
		print_epoch_status(model_stats, epoch)

		# Plot intermittent stuff after 10% increments
		has_improved_loss = test_loss < best_test_loss
		has_improved_tvalid =  test_tvalid > best_test_tvalid
		is_save_interval = (epoch % save_freq == 0)
		if has_improved_loss:
			best_test_loss = test_loss
		if has_improved_tvalid:
			best_test_tvalid = test_tvalid
		if has_improved_loss or has_improved_tvalid or is_save_interval:
			plot_stats(model_stats, epoch=epoch+1, output_path=output_path)
			model.plot_weights(n_epochs=epoch+1)
			model.make_traj_plots(all_target_states, all_predicted_states, all_rnn_predicted_residuals, all_hidden_states, name='train', epoch=epoch)
			model.make_traj_plots(target_sequence_synch, full_predicted_states_synch, rnn_predicted_residuals_synch, hidden_states_synch, name='test_synch', epoch=epoch)
			model.make_traj_plots(target_sequence_test, full_predicted_states_test, rnn_predicted_residuals_test, hidden_states_test, name='test', epoch=epoch)

		if is_save_interval:
			for name, val in model.named_parameters():
				if val.requires_grad:
					easy_name = model.lookup[name][0]
					print('|{0}|'.format(easy_name), np.linalg.norm(val.data))

	print('all done!')

def print_epoch_status(model_stats, epoch=-1):
	vals = {}
	vals['epoch'] = epoch
	vals['ltrain'] = np.median(model_stats['Train']['loss'][epoch,:])
	vals['ltest'] = np.median(model_stats['Test']['loss'][epoch,:])
	vals['ttrain'] = np.median(model_stats['Train']['t_valid'][epoch,:])
	vals['ttest'] = np.median(model_stats['Test']['t_valid'][epoch,:])
	status_string = 'Epoch {epoch}. l-train={ltrain}, l-test={ltest}, t-train={ttrain}, t-test={ttest}'.format(**vals)
	print(status_string)
	return


def plot_stats(model_stats, epoch=-1, output_path='.'):
	train_loss_vec = model_stats['Train']['loss']
	train_t_valid_vec = model_stats['Train']['t_valid']
	test_loss_vec = model_stats['Test']['loss']
	test_t_valid_vec = model_stats['Test']['t_valid']

	fig, ax_list = plt.subplots(2,1, figsize=[12,10], sharex=True)

	# loss function
	ax = ax_list[0]
	ax.errorbar(x=np.arange(epoch), y=np.mean(train_loss_vec[:epoch,:], axis=1), yerr=np.std(train_loss_vec[:epoch,:], axis=1), label='Training Loss', linestyle='-')
	ax.errorbar(x=np.arange(epoch), y=np.mean(test_loss_vec[:epoch,:], axis=1), yerr=np.std(test_loss_vec[:epoch,:], axis=1), label='Testing Loss', linestyle='--')
	ax.set_ylabel('Loss')
	# ax.set_xlabel('Epochs')
	ax.legend()

	# validity time
	ax = ax_list[1]
	ax.errorbar(x=np.arange(epoch), y=np.mean(train_t_valid_vec[:epoch,:], axis=1), yerr=np.std(train_t_valid_vec[:epoch,:], axis=1), label=' Train', linestyle='-')
	ax.errorbar(x=np.arange(epoch), y=np.mean(test_t_valid_vec[:epoch,:], axis=1), yerr=np.std(test_t_valid_vec[:epoch,:], axis=1), label=' Test', linestyle='--')
	ax.set_ylabel('Validity Time')
	ax.set_title('Validity Time')
	ax.legend()

	fig.suptitle('Train/Test Performance')
	fig.savefig(fname=os.path.join(output_path,'TrainTest_Performance'))
	plt.close(fig)
	return
