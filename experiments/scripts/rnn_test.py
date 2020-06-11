import os
import MARS_train_test as mars
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from time import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *
from rnn_models import *

import pandas as pd
import argparse

import pdb




parser = argparse.ArgumentParser(description='behaviorRNN')
parser.add_argument('--num_epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--lr', type=float, default=None, help='learning rate, whose default depends on the specified optimizer')
parser.add_argument('--num_frames', type=int, default=1000, help='number of frames per training video chunk')
parser.add_argument('--hidden_dim', type=int, default=10, help='number of dimensions for RNN hidden state')
parser.add_argument('--optimizer', type=str, default='SGD', help='specifiy which optimizer to use')
parser.add_argument('--loss', type=str, default='nn.MSELoss', help='specifiy which loss function to use')
parser.add_argument('--cell_type', type=str, default='LSTM', help='specifiy which RNN model to use')
parser.add_argument('--train_path', type=str, default='TRAIN_lite', help='specifiy path to TRAIN videos')
parser.add_argument('--test_path', type=str, default='TEST_lite', help='specifiy path to TEST videos')
parser.add_argument('--output_path', type=str, default='default_output', help='specifiy path to TEST videos')
parser.add_argument('--balance_weights', type=str2bool, default=True, help='If true, compute cost function weights based on relative class frequencies')
parser.add_argument('--use_gpu', type=str2bool, default=False, help='If true, use cuda')
parser.add_argument('--feature_style', type=str, default="keypoints_only", help='If true, set dtype=torch.cuda.FloatTensor and use cuda')
parser.add_argument('--stack_hidden', type=str2bool, default=True, help='include outputs from model as features.')
parser.add_argument('--learn_residual', type=str2bool, default=False, help='Learn RNN as residual of a given prediction')
parser.add_argument('--save_freq', type=int, default=1, help='interval of epochs for which we should save outputs')
parser.add_argument('--bidirectional', type=str2bool, default=False, help='interval of epochs for which we should save outputs')
parser.add_argument('--num_rnn_layers', type=int, default=1, help='number of layers of RNN cells')
FLAGS = parser.parse_args()


def main():
	output_path = FLAGS.output_path

	if FLAGS.use_gpu and not torch.cuda.is_available():
		# https://thedavidnguyenblog.xyz/installing-pytorch-1-0-stable-with-cuda-10-0-on-windows-10-using-anaconda/
		print('Trying to use GPU, but cuda is NOT AVAILABLE. Running with CPU instead.')
		FLAGS.use_gpu = False
		pdb.set_trace()

	# choose cuda-GPU or regular
	if FLAGS.use_gpu:
		dtype = torch.cuda.FloatTensor
		inttype = torch.cuda.LongTensor
	else:
		dtype = torch.FloatTensor
		inttype = torch.LongTensor

	if not os.path.exists(output_path):
		os.mkdir(output_path)

	settings_fname = output_path + '/run_settings.txt'
	write_settings(FLAGS, settings_fname)



	num_classes = ytrain[0].shape[1]

	input_dim = Xtrain[0].shape[1]
	model = get_model(input_dim=input_dim, hidden_dim=FLAGS.hidden_dim, num_layers=FLAGS.num_rnn_layers, cell_type=FLAGS.cell_type)

	if FLAGS.use_gpu:
		model.cuda()

	optimizer = get_optimizer(name=FLAGS.optimizer, params=model.parameters(), lr=FLAGS.lr)

	## Normalize the data
	Xtrain_stats = stats_of(Xtrain)
	Xtrain = normalize(X=Xtrain, stats=Xtrain_stats)
	Xtest = normalize(X=Xtest, stats=Xtrain_stats) # using Xtrain stats on purpose here...for now.

	loss_function = get_loss(name=FLAGS.loss)

	best_model_dict = {}

	# train the model
	num_frames = FLAGS.num_frames
	num_epochs = FLAGS.num_epochs

	train_loss_vec = np.zeros((num_epochs,1))
	test_loss_vec = np.zeros((num_epochs,1))

	for epoch in range(num_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
		if (epoch % FLAGS.save_freq)==0:
			do_saving = True
			make_plots = True
		else:
			do_saving = False
			make_plots = False
		t0 = time()
		# setence is our features, tags are INDICES of true label
		all_predicted_states = []
		all_target_states = []
		for v in range(len(Xtrain)):
			if FLAGS.learn_residual:
				big_input_bias = Xtrain_raw[v][:,glm_inds]
				big_input_bias = big_input_bias[:,reorder_glm_inds]
			big_input = Xtrain[v]
			big_target = ytrain[v]

			offset = np.random.randint(len(big_input) % num_frames - 1)
			permutations = list(range(int(len(big_input) / num_frames)))
			np.random.shuffle(permutations)
			for permutation in permutations:
				# sample a random chunk of video
				start_ind = permutation * num_frames + offset
				end_ind = start_ind + num_frames

				input_sequence = big_input[start_ind:end_ind,:]
				target_sequence = big_target[start_ind:end_ind,:]
				if FLAGS.learn_residual:
					bias_sequence = torch.FloatTensor(big_input_bias[start_ind:end_ind,:]).type(dtype)
				else:
					bias_sequence = 0
				# Step 1. Remember that Pytorch accumulates gradients.
				# We need to clear them out before each instance
				model.zero_grad()

				# Step 3. Run our forward pass.
				full_predicted_states, rnn_predicted_residuals = model(input_sequence=torch.FloatTensor(input_sequence).type(dtype),
												bias_sequence=bias_sequence).type(dtype)

				# Step 4. Compute the loss, gradients, and update the parameters by
				#  calling optimizer.step()
				loss = loss_function(predicted_class_states, target_sequence)
				loss.backward()
				optimizer.step()

				all_predicted_states.append(predicted_states)
				all_target_states.append(target_sequence)

		all_predicted_states = torch.cat(all_predicted_states)
		all_target_states = torch.cat(all_target_states)

		# Report Train losses after each epoch
		train_loss = loss_function(all_predicted_states, all_target_states)
		print('Epoch',epoch,' Train Loss=', train_loss.cpu().data.numpy().item())
		print('Train Epoch', epoch, time()-t0)

		# save data
		train_loss_vec[epoch] = train_loss.cpu().data.numpy().item()
		if do_saving:
			np.savetxt(output_path+'/train_loss_vec.txt',train_loss_vec[:(epoch+1)])

		# Report TEST performance after each epoch
		all_predicted_states = []
		all_target_states = []
		for v in range(len(Xtest)):
			if FLAGS.learn_residual:
				big_input_bias = Xtest_raw[v][:,glm_inds]
				big_input_bias = big_input_bias[:,reorder_glm_inds]
			big_input = Xtest[v]
			big_target = ytest[v]
			input_sequence = big_input
			target_sequence = big_target
			target_sequence = torch.tensor(np.argmax(target_sequence, axis=1)).type(inttype)
			if FLAGS.learn_residual:
				bias_sequence = torch.FloatTensor(big_input_bias).type(dtype)
			else:
				bias_sequence = 0

			# Step 3. Run our forward pass.
			predicted_class_states = model(input_state_sequence=torch.FloatTensor(input_sequence).type(dtype),
											input_prediction_sequence=bias_sequence).type(dtype)

			all_predicted_states.append(predicted_class_states)
			all_target_states.append(target_sequence)
		# Step 4. Compute the losses
		all_predicted_states = torch.cat(all_predicted_states)
		all_target_states = torch.cat(all_target_states)
		test_loss = loss_function(all_predicted_states, all_target_states)
		print('Epoch',epoch,' Test Loss=', test_loss.cpu().data.numpy().item())

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
			foo_test_loss = best_model_dict[model_nm]['Test'][FLAGS.loss]
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
			best_model_dict[model_nm]['Train'][FLAGS.loss] = train_loss_vec[my_ind].item()
			best_model_dict[model_nm]['Test'][FLAGS.loss] = test_loss_vec[my_ind].item()
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
			best_model_dict[model_nm]['Train'][FLAGS.loss] = train_loss_vec[my_ind].item()
			best_model_dict[model_nm]['Test'][FLAGS.loss] = test_loss_vec[my_ind].item()
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


		print('Test Epoch', epoch, time() - t0)


if __name__ == '__main__':
	main()


