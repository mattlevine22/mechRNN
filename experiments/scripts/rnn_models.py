import os, sys
import numpy as np
from time import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pdb




class RNN_VANILLA(nn.Module):

	def __init__(self, input_size, hidden_size, output_size, cell_type, embed_physics_prediction, use_physics_as_bias, dtype, t_synch, teacher_force_probability):
		super(RNN_VANILLA, self).__init__()
		self.teacher_force_probability = teacher_force_probability
		self.t_synch = t_synch
		self.dtype = dtype
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.embed_physics_prediction = embed_physics_prediction
		self.use_physics_as_bias = use_physics_as_bias

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

	def normalize(self, x):
		return x

	def unnormalize(self, x):
		return x

	def get_physics_prediction(self, x0, dt):
		# self.delta_t
		return x0

	def forward(self, input_state_sequence, physical_prediction_sequence=None, train=True):
		rnn_preds = [] #output of the RNN (i.e. residual)
		full_preds = [] #final output prediction (i.e. Psi0(x_t) + RNN(x_t,h_t))

		#useful link for teacher-forcing: https://towardsdatascience.com/lstm-for-time-series-prediction-de8aeb26f2ca
		#

		# input_states corresponds to data x_t
		# physical_prediction_sequence corresponds to predictions \Psi_0(x_t), i.e. \hat{x_t}
		# optional bias term driven by glm_scores

		if self.embed_physics_prediction:
			input_sequence = torch.stack(input_state_sequence, physical_prediction_sequence)
		else:
			input_sequence = input_state_sequence

		# initialize states (if doing batches)
		# h_t = torch.zeros((input_sequence.size(1), self.hidden_size), dtype=self.dtype)
		# c_t = torch.zeros((input_sequence.size(1), self.hidden_size), dtype=self.dtype)

		h_t = torch.zeros((input_sequence.size(0), self.hidden_size), dtype=self.dtype) # (batch, hidden_size)
		c_t = torch.zeros((input_sequence.size(0), self.hidden_size), dtype=self.dtype) # (batch, hidden_size)

		# consider Scheduled Sampling (https://arxiv.org/abs/1506.03099) where probability of using RNN-output increases as you train.
		for t in range(input_sequence.shape[1]):
			# get input to hidden state
			if train:
				# consider teacher forcing (using RNN output prediction as next training input instead of training data)
				if t>self.t_synch and random.random()<self.teacher_force_probability:
					input_t = full_pred #feed RNN prediction back in as next input
				else:
					input_t = input_sequence[:,t,:]
				physics_pred = self.get_physics_prediction(x0=self.unnormalize(input_t), dt=0.01)
			else:
				physics_pred = self.get_physics_prediction(x0=self.unnormalize(full_pred), dt=0.01)
				input_t = torch.stack(pred_output, self.normalize(physics_prediction))

			# evolve hidden state
			h_t, c_t = self.cell(input_t, (h_t, c_t)) # input needs to be dim (batch, input_size)
			rnn_pred = self.hidden2pred(h_t)
			full_pred = self.use_physics_as_bias * physics_pred + rnn_pred
			full_preds += [full_pred]
			rnn_preds += [rnn_pred]

		full_preds = torch.stack(full_preds, 1).squeeze(2)
		rnn_preds = torch.stack(rnn_preds, 1).squeeze(2)

		return full_preds, rnn_preds


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

def get_model(input_size, name='RNN_VANILLA', hidden_size=50, output_size=None, cell_type='LSTM', embed_physics_prediction=False, use_physics_as_bias=False, dtype=torch.float, t_synch=1000, teacher_force_probability=0.0):
	if output_size is None:
		output_size = input_size

	if name=='RNN_VANILLA':
		return RNN_VANILLA(input_size=input_size, hidden_size=hidden_size, output_size=output_size, cell_type=cell_type, dtype=dtype, embed_physics_prediction=embed_physics_prediction, use_physics_as_bias=use_physics_as_bias, t_synch=t_synch, teacher_force_probability=teacher_force_probability)
	else:
		return None


def train_RNN_new(y_noisy_train,
				y_noisy_test,
				output_dir='.',
				num_frames=500,
				num_epochs=10,
				save_freq=1,
				learn_residual=False,
				use_gpu=False,
				normz_info=None,
				**kwargs):

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
	Xtrain = y_noisy_train[None,:-1]
	ytrain = y_noisy_train[None,1:]

	Xtest = y_noisy_test[:-1]
	ytest = y_noisy_test[1:]

	# set up model
	model = get_model(input_size=Xtrain.shape[2])
	if use_gpu:
		model.cuda()
	optimizer = get_optimizer(params=model.parameters())
	loss_function = get_loss()

	## Normalize the data
	# Xtrain_stats = stats_of(Xtrain)
	# Xtrain = normalize(X=Xtrain, stats=Xtrain_stats)
	# Xtest = normalize(X=Xtest, stats=Xtrain_stats) # using Xtrain stats on purpose here...for now.


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
		if learn_residual:
			big_input_bias = Xtrain_raw[:,glm_inds]
			big_input_bias = big_input_bias[:,reorder_glm_inds]
		big_input = Xtrain
		big_target = ytrain

		offset = np.random.randint(big_input.shape[1] % num_frames - 1)
		permutations = list(range(int(big_input.shape[1] / num_frames)))
		np.random.shuffle(permutations)
		for permutation in permutations:
			# sample a random chunk of video
			start_ind = permutation * num_frames + offset
			end_ind = start_ind + num_frames

			input_state_sequence = big_input[:,start_ind:end_ind,:]
			target_sequence = torch.FloatTensor(big_target[:,start_ind:end_ind,:]).type(dtype)
			if learn_residual:
				bias_sequence = torch.FloatTensor(big_input_bias[:,start_ind:end_ind,:]).type(dtype)
			else:
				bias_sequence = 0
			# Step 1. Remember that Pytorch accumulates gradients.
			# We need to clear them out before each instance
			model.zero_grad()

			# Step 3. Run our forward pass.
			full_predicted_states, rnn_predicted_residuals = model(input_state_sequence=torch.FloatTensor(input_state_sequence).type(dtype),
											physical_prediction_sequence=bias_sequence)

			# Step 4. Compute the loss, gradients, and update the parameters by
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
		# print('Epoch',epoch,' Train Loss=', train_loss.cpu().data.numpy().item())
		# print('Train Epoch Length:', epoch, tim	e()-t0)

		# save data
		train_loss_vec[epoch] = train_loss.cpu().data.numpy().item()
		if do_saving:
			np.savetxt(output_path+'/train_loss_vec.txt',train_loss_vec[:(epoch+1)])

		# Report TEST performance after each epoch
		all_predicted_states = []
		all_target_states = []
		if learn_residual:
			big_input_bias = Xtest_raw[:,glm_inds]
			big_input_bias = big_input_bias[:,reorder_glm_inds]
		big_input = Xtest
		big_target = ytest
		input_state_sequence = big_input
		target_sequence = torch.FloatTensor(big_target).type(dtype)
		if learn_residual:
			bias_sequence = torch.FloatTensor(big_input_bias).type(dtype)
		else:
			bias_sequence = 0


		# Step 3. Run our forward pass.
		full_predicted_states, rnn_predicted_residuals = model(input_state_sequence=torch.FloatTensor(input_state_sequence).type(dtype),
										physical_prediction_sequence=bias_sequence)

		# Step 4. Compute the losses
		test_loss = loss_function(full_predicted_states, target_sequence)
		print('Epoch {epoch}. l-train={ltrain}, l-test={ltest}'.format(epoch=epoch, ltrain=train_loss.cpu().data.numpy().item(), ltest=test_loss.cpu().data.numpy().item()))

		# save data
		test_loss_vec[epoch] = test_loss.cpu().data.numpy().item()
		if do_saving:
			np.savetxt(output_path+'/test_loss_vec.txt',test_loss_vec[:(epoch+1)])

		# make plots
		if make_plots:
			# RNN performance plots
			predicted_states = all_predicted_states[:num_frames].cpu().data.numpy()
			actual = all_target_states[:num_frames].cpu().data.numpy()
			# fig = plot_predicted_vs_actual(predicted_states, actual, states = class_names)
			# # fig.suptitle('Train/Test Performance')
			# fig.savefig(fname=output_path+'/example_RNN_outputs')
			# plt.close(fig)


			prop_cycle = plt.rcParams['axes.prop_cycle']
			color_list = prop_cycle.by_key()['color']

			fig, ax_list = plt.subplots(2,1, figsize=[12,10], sharex=True)

			# loss function
			ax = ax_list[0]
			ax.plot(train_loss_vec[:(epoch+1)], label='Training Loss', linestyle='-')
			ax.plot(test_loss_vec[:(epoch+1)], label='Testing Loss', linestyle='--')
			ax.set_ylabel('Loss')
			# ax.set_xlabel('Epochs')
			ax.legend()

			# # precision
			ax = ax_list[1]
			ax.plot(train_t_valid_vec[:(epoch+1)], label=' Train', linestyle='-')
			ax.plot(test_t_valid_vec[:(epoch+1)], label=' Test', linestyle='--')
			ax.set_ylabel('Validity Time')
			ax.set_title('Validity Time')
			ax.legend()

			fig.suptitle('Train/Test Performance')
			fig.savefig(fname=output_path+'/TrainTest_Performance')
			plt.close(fig)


			## Now, choose the epoch that optimizes either Loss, Precision, or Recall and plot its performance
			fig, axlist = plt.subplots(1,3, figsize=[10,4], sharey=True)

			for ax in axlist:
				ax.yaxis.set_tick_params(labelleft=True)

			# plot GLM-alone performance
			cc = 0
			ax = axlist[cc]
			summary_list = []
			model_nm = 'Psi0'
			foo_test_loss = best_model_dict[model_nm]['Test']['loss']
			for c in range(num_classes):
				val_dict = best_model_dict[model_nm]['Test'][class_names[c]]
			df = pd.DataFrame(summary_list)
			sns.barplot(ax=ax, x='behavior', y='value', hue='metric', data=df)
			ax.xaxis.set_label_text("")
			ax.yaxis.set_label_text("")
			ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center', fontweight='light', fontsize='x-large')
			ax.set_title('GLM performance (Test Loss = {0:.2f})'.format(foo_test_loss))


			cc = 1
			ax = axlist[cc]
			my_ind = np.argmin(test_loss_vec[:(epoch+1)])
			summary_list = []
			model_nm = 'RNN_best_test_loss'
			if model_nm not in best_model_dict:
				best_model_dict[model_nm] = {'Train':{}, 'Test':{}}
			best_model_dict[model_nm]['Train']['loss'] = train_loss_vec[my_ind].item()
			best_model_dict[model_nm]['Test']['loss'] = test_loss_vec[my_ind].item()
			for c in range(num_classes):
				cnm = class_names[c]
				if cnm not in best_model_dict[model_nm]['Train']:
					best_model_dict[model_nm]['Train'][cnm] = {}
					best_model_dict[model_nm]['Test'][cnm] = {}
				best_model_dict[model_nm]['Train'][cnm]['Precision'] = train_precision_vec[my_ind,c].item()
				best_model_dict[model_nm]['Train'][cnm]['Recall'] = train_recall_vec[my_ind,c].item()
				best_model_dict[model_nm]['Test'][cnm]['Precision'] = test_precision_vec[my_ind,c].item()
				best_model_dict[model_nm]['Test'][cnm]['Recall'] = test_recall_vec[my_ind,c].item()
				pred_dict = {'behavior': cnm, 'metric': 'Precision', 'value': test_precision_vec[my_ind,c].item()}
				recall_dict = {'behavior': cnm, 'metric': 'Recall', 'value': test_recall_vec[my_ind,c].item()}
				summary_list.append(pred_dict)
				summary_list.append(recall_dict)
			df = pd.DataFrame(summary_list)
			sns.barplot(ax=ax, x='behavior', y='value', hue='metric', data=df)
			ax.xaxis.set_label_text("")
			ax.yaxis.set_label_text("")
			ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center', fontweight='light', fontsize='x-large')
			ax.set_title('Best Test Loss (Test Loss = {0:.2f})'.format(test_loss_vec[my_ind].item()))

			cc = 2
			ax = axlist[cc]
			my_ind = np.argmin(train_loss_vec[:(epoch+1)])
			summary_list = []
			model_nm = 'RNN_best_train_loss'
			if model_nm not in best_model_dict:
				best_model_dict[model_nm] = {'Train':{}, 'Test':{}}
			best_model_dict[model_nm]['Train']['loss'] = train_loss_vec[my_ind].item()
			best_model_dict[model_nm]['Test']['loss'] = test_loss_vec[my_ind].item()
			for c in range(num_classes):
				cnm = class_names[c]
				if cnm not in best_model_dict[model_nm]['Train']:
					best_model_dict[model_nm]['Train'][cnm] = {}
					best_model_dict[model_nm]['Test'][cnm] = {}
				best_model_dict[model_nm]['Train'][cnm]['Precision'] = train_precision_vec[my_ind,c].item()
				best_model_dict[model_nm]['Train'][cnm]['Recall'] = train_recall_vec[my_ind,c].item()
				best_model_dict[model_nm]['Test'][cnm]['Precision'] = test_precision_vec[my_ind,c].item()
				best_model_dict[model_nm]['Test'][cnm]['Recall'] = test_recall_vec[my_ind,c].item()
				pred_dict = {'behavior': cnm, 'metric': 'Precision', 'value': test_precision_vec[my_ind,c].item()}
				recall_dict = {'behavior': cnm, 'metric': 'Recall', 'value': test_recall_vec[my_ind,c].item()}
				summary_list.append(pred_dict)
				summary_list.append(recall_dict)
			df = pd.DataFrame(summary_list)
			sns.barplot(ax=ax, x='behavior', y='value', hue='metric', data=df)
			ax.xaxis.set_label_text("")
			ax.yaxis.set_label_text("")
			ax.set_xticklabels(ax.get_xticklabels(), rotation=0, horizontalalignment='center', fontweight='light', fontsize='x-large')
			ax.set_title('Best Train Loss (Test Loss = {0:.2f})'.format(test_loss_vec[my_ind].item()))

			fig.subplots_adjust(bottom=0.1)
			fig.suptitle('Model Test Performances')
			fig.savefig(fname=output_path+'/BarChart_Performance')
			plt.close(fig)

			# write out best model summary
			with open(best_model_fname, 'w') as f:
				json.dump(best_model_dict, f, indent=2)

		# print('Test Epoch', epoch, time() - t0)
