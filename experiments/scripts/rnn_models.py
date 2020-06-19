import os, sys
import numpy as np
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from utils import traj_div_time
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
		train_RNN_new(**setts)
		# train_chaosRNN(**setts)

	print('Ran training in:', time()-t0)
	return

class RNN_VANILLA(nn.Module):

	def __init__(self, input_size, hidden_size, output_size, cell_type, embed_physics_prediction, use_physics_as_bias, dtype, t_synch, teacher_force_probability, norm_dict, ode_params, ode):
		super(RNN_VANILLA, self).__init__()
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
		elif cell_type=='GRU':
			self.cell = nn.GRUCell(input_size, hidden_size)
		else:
			try:
				self.cell = nn.RNNCell(input_size, hidden_size, nonlinearity=cell_type)
			except:
				self.cell = nn.RNNCell(input_size, hidden_size, nonlinearity='relu')

		# The linear layer that maps from hidden state space to tag space
		self.hidden2pred = nn.Linear(hidden_size, output_size)

	def normalize(self, y):
		return normalize(norm_dict=self.norm_dict, y=y)

	def unnormalize(self, y_norm):
		return unnormalize(norm_dict=self.norm_dict, y_norm=y_norm)

	def get_physics_prediction(self, ic, solver_failed=False):
		#input and output are unnormalized
		n_ics = ic.shape[0]

		y_pred = np.zeros(ic.shape)
		for c in range(n_ics):
			# check if y0 is a bad initial condition
			try:
				if (any(abs(ic[c])>1000)):
					print('ODE initial conditions are huge, so not even trying to solve the system. Applying the Identity forward map instead.',y0)
					solver_failed = True
			except:
				pdb.set_trace()

			if not solver_failed:
				sol = solve_ivp(fun=lambda t, y: self.ode.rhs(y, t), t_span=self.tspan, y0=ic[c], t_eval=self.t_eval, **self.ode_params)
				# sol = solve_ivp(fun=lambda t, y: model(y, t, *model_params['ode_params']), t_span=(self.tspan[0], self.tspan[-1]), y0=y0.T, method=model_params['ode_int_method'], rtol=model_params['ode_int_rtol'], atol=model_params['ode_int_atol'], max_step=model_params['ode_int_max_step'], t_eval=self.tspan)
				y_out = sol.y.T
				if not sol.success:
					# solver failed
					print('ODE solver has failed at ic=',ic[c])
					solver_failed = True

			if solver_failed:
				y_pred[c] = np.copy(ic[c].numpy()) # persist previous solution
			else:
				# solver is OKAY--use the solution like a good boy!
				y_pred[c] = y_out[-1,:]

		return torch.FloatTensor(y_pred).type(self.dtype)

	def forward(self, input_state_sequence, n_steps=None, physical_prediction_sequence=None, train=True, synch_mode=False, h_t=None, c_t=None):
		# input_state_sequence should be normalized
		# physical_prediction_sequence should be normalized

		if n_steps is None:
			n_steps = input_state_sequence.shape[1]

		rnn_preds = [] #output of the RNN (i.e. residual)
		full_preds = [] #final output prediction (i.e. Psi0(x_t) + RNN(x_t,h_t))

		#useful link for teacher-forcing: https://towardsdatascience.com/lstm-for-time-series-prediction-de8aeb26f2ca
		#

		if h_t is None or c_t is None:
			h_t = torch.zeros((input_state_sequence.size(0), self.hidden_size), dtype=self.dtype) # (batch, hidden_size)
			c_t = torch.zeros((input_state_sequence.size(0), self.hidden_size), dtype=self.dtype) # (batch, hidden_size)

		full_rnn_pred = input_state_sequence[:,0] #normalized
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
					# if np.isnan(input_t.detach().numpy()).any():
					# 	pdb.set_trace()
					physics_pred = self.normalize(self.get_physics_prediction(ic=self.unnormalize(input_t).detach().numpy()))

				if self.embed_physics_prediction:
					input_t = torch.stack(input_t, physics_pred)
			else:
				physics_pred = 0

			# evolve hidden state
			h_t, c_t = self.cell(input_t, (h_t, c_t)) # input needs to be dim (batch, input_size)
			rnn_pred = self.hidden2pred(h_t)
			full_rnn_pred = self.use_physics_as_bias * physics_pred + rnn_pred # unnormalized
			full_preds += [full_rnn_pred]
			rnn_preds += [rnn_pred]

		full_preds = torch.stack(full_preds, 1).squeeze(2)
		rnn_preds = torch.stack(rnn_preds, 1).squeeze(2)

		return full_preds, rnn_preds, h_t, c_t


def get_optimizer(params, name='SGD', lr=None):
	if name=='SGD':
		if lr is None:
			lr = 0.05
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

def get_model(input_size, name='RNN_VANILLA', hidden_size=50, output_size=None, cell_type='LSTM', embed_physics_prediction=False, use_physics_as_bias=False, dtype=torch.float, t_synch=1000, teacher_force_probability=0.0, norm_dict=None, ode_params=None, ode=None):
	if output_size is None:
		output_size = input_size

	if name=='RNN_VANILLA':
		return RNN_VANILLA(ode_params=ode_params, input_size=input_size, hidden_size=hidden_size, output_size=output_size, cell_type=cell_type, dtype=dtype, embed_physics_prediction=embed_physics_prediction, use_physics_as_bias=use_physics_as_bias, t_synch=t_synch, teacher_force_probability=teacher_force_probability, norm_dict=norm_dict, ode=ode)
	else:
		return None

def train_RNN_new(y_noisy_train,
				y_noisy_test,
				y_noisy_testSynch,
				model_params=None,
				output_dir='.',
				num_frames=500,
				num_epochs=10,
				save_freq=1,
				use_physics_as_bias=False,
				use_gpu=False,
				normz_info=None,
				ODE=None,
				**kwargs):

	n_test_traj = y_noisy_test.shape[0]
	n_train_traj = 1 #y_noisy_train.shape[0]

	model_stats = {'Train': {'loss': np.zeros((num_epochs,n_train_traj)),
							't_valid': np.zeros((num_epochs,n_train_traj))
							},
					'Test': {'loss': np.zeros((num_epochs,n_test_traj)),
							't_valid': np.zeros((num_epochs,n_test_traj))
							}
						}

	output_path = output_dir
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

	if not os.path.exists(output_path):
		os.mkdir(output_path)

	# get data
	Xtrain = y_noisy_train[None,:-1,:]
	ytrain = y_noisy_train[None,1:,:]
	Xtest = y_noisy_test[:,:-1,:]
	ytest = y_noisy_test[:,1:,:]
	Xtest_synch = y_noisy_testSynch[:,:-1,:]
	ytest_synch = y_noisy_testSynch[:,1:,:]

	# get unnormalized data
	Xtrain_raw = unnormalize(norm_dict=normz_info, y_norm=Xtrain)
	ytrain_raw = unnormalize(norm_dict=normz_info, y_norm=ytrain)
	Xtest_raw = unnormalize(norm_dict=normz_info, y_norm=Xtest)
	ytest_raw = unnormalize(norm_dict=normz_info, y_norm=ytest)

	Xtest_synch_raw = unnormalize(norm_dict=normz_info, y_norm=Xtest_synch)
	ytest_synch_raw = unnormalize(norm_dict=normz_info, y_norm=ytest_synch)

	# set up model
	model = get_model(ode_params=model_params, input_size=Xtrain.shape[2], norm_dict=normz_info, use_physics_as_bias=use_physics_as_bias, ode=ODE)
	if use_gpu:
		model.cuda()
	optimizer = get_optimizer(params=model.parameters())
	loss_function = get_loss()

	best_model_dict = {}

	# train the model
	train_loss_vec = np.zeros((num_epochs,1))
	test_loss_vec = np.zeros((num_epochs,1))

	for epoch in range(num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
		if (epoch % save_freq)==0:
			do_saving = True
			make_plots = True
		else:
			do_saving = False
			make_plots = False

		make_plots = False
		do_saving = False

		t0 = time()
		# setence is our features, tags are INDICES of true label
		all_predicted_states = []
		all_target_states = []

		offset = np.random.randint(Xtrain.shape[1] % num_frames - 1)
		permutations = list(range(int(Xtrain.shape[1] / num_frames)))
		np.random.shuffle(permutations)
		for permutation in permutations:
			# sample a random chunk of video
			start_ind = permutation * num_frames + offset
			end_ind = start_ind + num_frames

			if use_physics_as_bias:
				bias_sequence = torch.FloatTensor(Xtrain[:,start_ind:end_ind,:]).type(dtype) #normalized
			else:
				bias_sequence = None
			# Step 1. Remember that Pytorch accumulates gradients.
			# We need to clear them out before each instance
			model.zero_grad()

			# Step 3. Run our forward pass.
			full_predicted_states, rnn_predicted_residuals, h_t, c_t = model(input_state_sequence=torch.FloatTensor(Xtrain[:,start_ind:end_ind,:]).type(dtype),
											physical_prediction_sequence=bias_sequence, train=True)

			# unnormalize the data
			target_sequence = torch.FloatTensor(ytrain_raw[:,start_ind:end_ind,:]).type(dtype)
			full_predicted_states = model.unnormalize(full_predicted_states)
			rnn_predicted_residuals = model.unnormalize(rnn_predicted_residuals)

			# Step 4: Plot training evaluation
			# make_train_plot(target_sequence, full_predicted_states, rnn_predicted_residuals, h_t, c_t)
			make_traj_plots(target_sequence, full_predicted_states, rnn_predicted_residuals, h_t, c_t, name='train', epoch=epoch)

			# Step 5. Compute the loss, gradients, and update the parameters by
			#  calling optimizer.step()
			loss = loss_function(full_predicted_states, target_sequence)
			loss.backward()
			optimizer.step()

			all_predicted_states.append(full_predicted_states)
			all_target_states.append(target_sequence)

		all_predicted_states = torch.cat(all_predicted_states, 1)
		all_target_states = torch.cat(all_target_states, 1)

		# Report Train losses after each epoch
		train_loss = loss_function(all_predicted_states, all_target_states)
		for c in range(n_train_traj):
			model_stats['Test']['t_valid'][epoch,c] = traj_div_time(Xtrue=all_target_states[c,:,:], Xpred=all_predicted_states[c,:,:], delta_t=model.delta_t, avg_output=model.time_avg_norm)
			model_stats['Train']['loss'][epoch,c] = loss_function(all_predicted_states[c,:,:], all_target_states[c,:,:]).cpu().data.numpy().item()

		# print('Epoch',epoch,' Train Loss=', train_loss.cpu().data.numpy().item())
		# print('Train Epoch Length:', epoch, tim	e()-t0)

		# save data
		# if do_saving:
		# 	np.savetxt(output_path+'/train_loss_vec.txt',model_stats['Train']['loss'][:(epoch+1)])

		### Report TEST performance after each epoch

		# Step 1. Run forward pass with synchronization data
		full_predicted_states, rnn_predicted_residuals, h_t, c_t = model(input_state_sequence=torch.FloatTensor(Xtest_synch).type(dtype),
										physical_prediction_sequence=None, train=False, synch_mode=True)
		# unnormalize the data
		target_sequence = torch.FloatTensor(ytest_synch_raw).type(dtype)
		full_predicted_states = model.unnormalize(full_predicted_states)
		rnn_predicted_residuals = model.unnormalize(rnn_predicted_residuals)
		make_traj_plots(target_sequence, full_predicted_states, rnn_predicted_residuals, h_t, c_t, name='test_synch', epoch=epoch)

		# Step 2. Run our forward pass with synchronized RNN
		# Note that bias-terms and physical predictions must be computed on the fly
		full_predicted_states, rnn_predicted_residuals, h_t, c_t = model(input_state_sequence=model.normalize(full_predicted_states[:,-1,:]),
										n_steps = Xtest.shape[1],
										physical_prediction_sequence=None, train=False, synch_mode=False,
										h_t=h_t, c_t=c_t)

		# unnormalize the data
		target_sequence = torch.FloatTensor(ytest_raw).type(dtype)
		full_predicted_states = model.unnormalize(full_predicted_states)
		rnn_predicted_residuals = model.unnormalize(rnn_predicted_residuals)
		make_traj_plots(target_sequence, full_predicted_states, rnn_predicted_residuals, h_t, c_t, name='test', epoch=epoch)

		# Step 3. Compute the losses
		test_loss = loss_function(full_predicted_states, target_sequence)
		for c in range(n_test_traj):
			model_stats['Test']['t_valid'][epoch,c] = traj_div_time(Xtrue=target_sequence[c,:,:], Xpred=full_predicted_states[c,:,:], delta_t=model.delta_t, avg_output=model.time_avg_norm)
			model_stats['Test']['loss'][epoch,c] = loss_function(full_predicted_states[c,:,:], torch.FloatTensor(ytest[c,:,:]).type(dtype)).cpu().data.numpy().item()

		print_epoch_status(model_stats, epoch)

		# Step 4. Plot stats of Train/Test performance
		plot_stats(model_stats, epoch=epoch+1, output_path=output_path)

		# Step 5. save data
		# if do_saving:
		# 	np.savetxt(output_path+'/test_loss_vec.txt',model_stats['Test']['loss'][:(epoch+1)])

		# print('Test Epoch', epoch, time() - t0)

def make_traj_plots(Xtrue, Xpred, Xpred_residuals, h_t, c_t, name, epoch):
	return

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
	fig.savefig(fname=output_path+'/TrainTest_Performance')
	plt.close(fig)
	return
