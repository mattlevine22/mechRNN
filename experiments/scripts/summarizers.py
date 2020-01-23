import os
from time import time
from datetime import timedelta
import math
import numpy as np
import numpy.matlib
# from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import pdb

def extract_epsilon_performance(my_dirs, output_fname="./epsilon_comparisons", win=1, many_epochs=True, eps_token='epsBadness'):
	t_vec = [1,2,4,6,8,10]
	n_gprs = 4
	# first, get sizes of things...max window size is 10% of whole test set.
	init_dirs = [x for x in my_dirs if 'RNN' in x.split('/')[-1]]
	d_label = my_dirs[0].split("/")[-1].rstrip('_noisy').rstrip('_clean')
	x_test = pd.DataFrame(np.loadtxt(init_dirs[0]+"/loss_vec_clean_test.txt"))
	n_vals = x_test.shape[0]

	win = min(win,n_vals//3)
	model_performance = {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}}
	rnn_performance = {True: {'mse':(), 't_valid':(), 'mse_time':(), 't_valid_time':{}},
					   False: {'mse':(), 't_valid':(), 'mse_time':(), 't_valid_time':{}}}
	hybrid_performance = {True: {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}},
					      False: {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}}}
	gpr_performance = [{'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}} for _ in range(n_gprs)]
	for d in my_dirs:
		d_label = d.split("/")[-1].rstrip('_noisy').rstrip('_clean')

		if 'residualFalse' in d_label:
			is_resid = False
		elif 'residualTrue' in d_label:
			is_resid = True
		elif 'GPR' in d_label:
			is_resid = True
		elif 'RNN':
			# assume for now that if residual not specified, then is_resid=False
			is_resid = False
		else:
			print('Uh oh, unrecognized d_label')
			print(d_label)

		if 'vanilla' in d_label:
			my_eps = None
		else:
			my_eps = float([z.strip(eps_token) for z in d_label.split('_') if eps_token in z][-1])
			# do i need to force ndmin?
			model_loss = np.loadtxt(d+'/perfectModel_loss_vec_clean_test.txt',ndmin=1)
			model_t_valid = np.loadtxt(d+'/perfectModel_prediction_validity_time_clean_test.txt',ndmin=1)
			for kkt in range(model_loss.shape[0]):
				if my_eps in model_performance['mse']:
					model_performance['mse'][my_eps] += (float(model_loss[kkt]),)
					model_performance['t_valid'][my_eps] += (float(model_t_valid[kkt]),)
				else:
					model_performance['mse'][my_eps] = (float(model_loss[kkt]),)
					model_performance['t_valid'][my_eps] = (float(model_t_valid[kkt]),)

		# do i need to force ndmin?
		x_train = pd.DataFrame(np.loadtxt(d+"/loss_vec_train.txt"))
		x_test = pd.DataFrame(np.loadtxt(d+"/loss_vec_clean_test.txt"))
		gpr_valid_test = [np.loadtxt(d+'/GPR{0}_prediction_validity_time_clean_test.txt'.format(gp+1),ndmin=1) for gp in range(n_gprs)]
		gpr_test = [np.loadtxt(d+'/GPR{0}_loss_vec_clean_test.txt'.format(gp+1),ndmin=1) for gp in range(n_gprs)]
		# gpr2_test = np.loadtxt(d+'/GPR2_loss_vec_clean_test.txt',ndmin=1)
		# gpr2_valid_test = np.loadtxt(d+'/GPR2_prediction_validity_time_clean_test.txt',ndmin=1)

		if many_epochs:
			x_valid_test = pd.DataFrame(np.loadtxt(d+"/prediction_validity_time_clean_test.txt"))
			# x_kl_test = pd.DataFrame(np.loadtxt(d+"/kl_vec_inv_clean_test.txt"))

		n_tests = x_valid_test.shape[1]
		for kkt in range(n_tests):
			x_train = x_train.rolling(win).mean()
			x_test.loc[:,kkt] = x_test.loc[:,kkt].rolling(win).mean()
			if many_epochs:
				x_valid_test.loc[:,kkt] = x_valid_test.loc[:,kkt].rolling(win).mean()
				# for kk in plot_state_indices:
				# 	ax4.plot(x_kl_test.loc[:,kk].rolling(win).mean(), label=d_label)
			if my_eps is None:
				rnn_performance[is_resid]['mse'] += (float(np.min(x_test.loc[:,kkt])),)
				# rnn_performance[is_resid]['mse_time'] += (float(np.nanargmin(x_test)),)
				if many_epochs:
					rnn_performance[is_resid]['t_valid'] += (float(np.max(x_valid_test.loc[:,kkt])),)
					for tt in t_vec:
						if tt in rnn_performance[is_resid]['t_valid_time']:
							try:
								rnn_performance[is_resid]['t_valid_time'][tt] += (x_valid_test.loc[x_valid_test.loc[:,kkt]>tt,kkt].index[0],)
							except:
								rnn_performance[is_resid]['t_valid_time'][tt] += (np.inf,)
						else:
							try:
								rnn_performance[is_resid]['t_valid_time'][tt] = (x_valid_test.loc[x_valid_test.loc[:,kkt]>tt,kkt].index[0],)
							except:
								rnn_performance[is_resid]['t_valid_time'][tt] = (np.inf,)
			elif my_eps in hybrid_performance[is_resid]['mse']:
				hybrid_performance[is_resid]['mse'][my_eps] += (float(np.min(x_test.loc[:,kkt])),)
				for gp in range(n_gprs):
					gpr_performance[gp]['mse'][my_eps] += (float(np.min(gpr_test[gp][kkt])),)
				# gpr1_performance['mse'][my_eps] += (float(np.min(gpr1_test[kkt])),)
				# gpr2_performance['mse'][my_eps] += (float(np.min(gpr2_test[kkt])),)
				# for tt in t_vec:
				# 	hybrid_performance[is_resid]['mse_time'][my_eps][tt] += (x_test[x_test.iloc[:,0]>tt].index[0],)
				if many_epochs:
					hybrid_performance[is_resid]['t_valid'][my_eps] += (float(np.max(x_valid_test.loc[:,kkt])),)
					for gp in range(n_gprs):
						gpr_performance[gp]['t_valid'][my_eps] += (float(np.max(gpr_valid_test[gp][kkt])),)
					# gpr1_performance['t_valid'][my_eps] += (float(np.max(gpr1_valid_test[kkt])),)
					# gpr2_performance['t_valid'][my_eps] += (float(np.max(gpr2_valid_test[kkt])),)
					for tt in t_vec:
						try:
							hybrid_performance[is_resid]['t_valid_time'][my_eps][tt] += (x_valid_test.loc[x_valid_test.loc[:,kkt]>tt,kkt].index[0],)
							# gpr1_performance['t_valid_time'][my_eps][tt] += (gpr1_valid_test.loc[gpr1_valid_test.loc[:,kkt]>tt,kkt].index[0],)
							# gpr2_performance['t_valid_time'][my_eps][tt] += (gpr2_valid_test.loc[gpr2_valid_test.loc[:,kkt]>tt,kkt].index[0],)
						except:
							hybrid_performance[is_resid]['t_valid_time'][my_eps][tt] += (np.inf,)
							for gp in range(n_gprs):
								gpr_performance[gp]['t_valid_time'][my_eps][tt] += (np.inf,)
							# gpr1_performance['t_valid_time'][my_eps][tt] += (np.inf,)
							# gpr2_performance['t_valid_time'][my_eps][tt] += (np.inf,)
			else:
				hybrid_performance[is_resid]['mse'][my_eps] = (float(np.min(x_test.loc[:,kkt])),)
				for gp in range(n_gprs):
					gpr_performance[gp]['mse'][my_eps] = (float(np.min(gpr_test[gp][kkt])),)
				# gpr1_performance['mse'][my_eps] = (float(np.min(gpr1_test[kkt])),)
				# gpr2_performance['mse'][my_eps] = (float(np.min(gpr2_test[kkt])),)
				# hybrid_performance[is_resid]['mse_time'][my_eps] = {}
				# for tt in t_vec:
				# 	hybrid_performance[is_resid]['mse_time'][my_eps][tt] = (x_test[x_test.iloc[:,0]>tt].index[0],)
				if many_epochs:
					hybrid_performance[is_resid]['t_valid'][my_eps] = (float(np.max(x_valid_test.loc[:,kkt])),)
					# gpr1_performance['t_valid'][my_eps] = (float(np.max(gpr1_valid_test[kkt])),)
					# gpr2_performance['t_valid'][my_eps] = (float(np.max(gpr2_valid_test[kkt])),)
					hybrid_performance[is_resid]['t_valid_time'][my_eps] = {}
					# gpr1_performance['t_valid_time'][my_eps] = {}
					# gpr2_performance['t_valid_time'][my_eps] = {}

					for gp in range(n_gprs):
						gpr_performance[gp]['t_valid'][my_eps] = (float(np.max(gpr_valid_test[gp][kkt])),)
						gpr_performance[gp]['t_valid_time'][my_eps] = {}


					for tt in t_vec:
						try:
							hybrid_performance[is_resid]['t_valid_time'][my_eps][tt] += (x_valid_test.loc[x_valid_test.loc[:,kkt]>tt,kkt].index[0],)
							# gpr1_performance['t_valid_time'][my_eps][tt] += (gpr1_valid_test.loc[gpr1_valid_test.loc[:,kkt]>tt,kkt].index[0],)
							# gpr2_performance['t_valid_time'][my_eps][tt] += (gpr2_valid_test.loc[gpr2_valid_test.loc[:,kkt]>tt,kkt].index[0],)
						except:
							hybrid_performance[is_resid]['t_valid_time'][my_eps][tt] = (np.inf,)
							for gp in range(n_gprs):
								gpr_performance[gp]['t_valid_time'][my_eps][tt] = (np.inf,)
							# gpr1_performance['t_valid_time'][my_eps][tt] = (np.inf,)
							# gpr2_performance['t_valid_time'][my_eps][tt] = (np.inf,)

	# now summarize
	ode_test_loss_mins = {key: np.min(model_performance['mse'][key]) for key in model_performance['mse']}
	ode_test_loss_maxes = {key: np.max(model_performance['mse'][key]) for key in model_performance['mse']}
	ode_test_loss_means = {key: np.mean(model_performance['mse'][key]) for key in model_performance['mse']}
	ode_test_loss_medians = {key: np.median(model_performance['mse'][key]) for key in model_performance['mse']}
	ode_test_loss_stds = {key: np.std(model_performance['mse'][key]) for key in model_performance['mse']}

	gpr_test_loss_mins = []
	gpr_test_loss_maxes = []
	gpr_test_loss_means = []
	gpr_test_loss_medians = []
	gpr_test_loss_stds = []
	for gp in range(n_gprs):
		gpr_test_loss_mins.append({key: np.min(gpr_performance[gp]['mse'][key]) for key in gpr_performance[gp]['mse']})
		gpr_test_loss_maxes.append({key: np.max(gpr_performance[gp]['mse'][key]) for key in gpr_performance[gp]['mse']})
		gpr_test_loss_means.append({key: np.mean(gpr_performance[gp]['mse'][key]) for key in gpr_performance[gp]['mse']})
		gpr_test_loss_medians.append({key: np.median(gpr_performance[gp]['mse'][key]) for key in gpr_performance[gp]['mse']})
		gpr_test_loss_stds.append({key: np.std(gpr_performance[gp]['mse'][key]) for key in gpr_performance[gp]['mse']})


	test_loss_medians = {}
	test_loss_stds = {}
	rnn_test_loss_medians = {}
	rnn_test_loss_stds = {}
	for is_resid in [True, False]:
		# test_loss_mins[is_resid] = {key: np.min(hybrid_performance['mse'][key]) for key in hybrid_performance['mse']}
		# test_loss_maxes[is_resid] = {key: np.max(hybrid_performance['mse'][key]) for key in hybrid_performance['mse']}
		# test_loss_means[is_resid] = {key: np.mean(hybrid_performance['mse'][key]) for key in hybrid_performance['mse']}
		test_loss_medians[is_resid] = {key: np.median(hybrid_performance[is_resid]['mse'][key]) for key in hybrid_performance[is_resid]['mse']}
		test_loss_stds[is_resid] = {key: np.std(hybrid_performance[is_resid]['mse'][key]) for key in hybrid_performance[is_resid]['mse']}

		# rnn_test_loss_mins[is_resid] = np.min(rnn_performance[is_resid]['mse'])
		# rnn_test_loss_maxes[is_resid] = np.max(rnn_performance[is_resid]['mse'])
		# rnn_test_loss_means[is_resid] = np.mean(rnn_performance[is_resid]['mse'])
		rnn_test_loss_medians[is_resid] = np.median(rnn_performance[is_resid]['mse'])
		rnn_test_loss_stds[is_resid] = np.std(rnn_performance[is_resid]['mse'])

	# plot summary
	fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,
		figsize = [10, 10],
		sharey=False, sharex=False)

	eps_vec = sorted(test_loss_medians[True].keys())

	try:
		median_vec = [ode_test_loss_medians[eps] for eps in eps_vec]
		std_vec = [ode_test_loss_stds[eps] for eps in eps_vec]
		ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='model-only', color='red')
	except:
		pass

	for gp in range(n_gprs):
		if gp==3:
			gp_nm = 'vanilla GPR'
		else:
			gp_nm = 'hybrid GPR {0}'.format(gp+1)
		# hybrid GPR
		median_vec = [gpr_test_loss_medians[gp][eps] for eps in eps_vec]
		std_vec = [gpr_test_loss_stds[gp][eps] for eps in eps_vec]
		ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label=gp_nm, linestyle='--')

	# # hybrid GPR 2
	# median_vec = [gpr2_test_loss_medians[eps] for eps in eps_vec]
	# std_vec = [gpr2_test_loss_stds[eps] for eps in eps_vec]
	# ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid GPR 2', color='green', linestyle='--')

	for is_resid in [True,False]:
		# hybrid RNN
		median_vec = [test_loss_medians[is_resid][eps] for eps in eps_vec]
		std_vec = [test_loss_stds[is_resid][eps] for eps in eps_vec]
		ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid RNN'+is_resid*' residual', color='blue')

		try:
			ax1.errorbar(x=eps_vec, y=[rnn_test_loss_medians[is_resid]]*len(eps_vec), yerr=[rnn_test_loss_stds[is_resid]]*len(eps_vec) ,label='vanilla RNN'+is_resid*' residual', color='black')
		except:
			pass

	ax1.set_xlabel('epsilon Model Error')
	ax1.set_ylabel('Test Loss (MSE)')
	ax1.legend()

	if many_epochs:
		ode_t_valid_mins = {key: np.min(model_performance['t_valid'][key]) for key in model_performance['t_valid']}
		ode_t_valid_maxes = {key: np.max(model_performance['t_valid'][key]) for key in model_performance['t_valid']}
		ode_t_valid_means = {key: np.mean(model_performance['t_valid'][key]) for key in model_performance['t_valid']}
		ode_t_valid_medians = {key: np.median(model_performance['t_valid'][key]) for key in model_performance['t_valid']}
		ode_t_valid_stds = {key: np.std(model_performance['t_valid'][key]) for key in model_performance['t_valid']}

		gpr_t_valid_mins = []
		gpr_t_valid_maxes = []
		gpr_t_valid_means = []
		gpr_t_valid_medians = []
		gpr_t_valid_stds = []
		for gp in range(n_gprs):
			gpr_t_valid_mins.append({key: np.min(gpr_performance[gp]['t_valid'][key]) for key in gpr_performance[gp]['t_valid']})
			gpr_t_valid_maxes.append({key: np.max(gpr_performance[gp]['t_valid'][key]) for key in gpr_performance[gp]['t_valid']})
			gpr_t_valid_means.append({key: np.mean(gpr_performance[gp]['t_valid'][key]) for key in gpr_performance[gp]['t_valid']})
			gpr_t_valid_medians.append({key: np.median(gpr_performance[gp]['t_valid'][key]) for key in gpr_performance[gp]['t_valid']})
			gpr_t_valid_stds.append({key: np.std(gpr_performance[gp]['t_valid'][key]) for key in gpr_performance[gp]['t_valid']})

		# gpr1_t_valid_mins = {key: np.min(gpr1_performance['t_valid'][key]) for key in gpr1_performance['t_valid']}
		# gpr1_t_valid_maxes = {key: np.max(gpr1_performance['t_valid'][key]) for key in gpr1_performance['t_valid']}
		# gpr1_t_valid_medians = {key: np.median(gpr1_performance['t_valid'][key]) for key in gpr1_performance['t_valid']}
		# gpr1_t_valid_means = {key: np.mean(gpr1_performance['t_valid'][key]) for key in gpr1_performance['t_valid']}
		# gpr1_t_valid_stds = {key: np.std(gpr1_performance['t_valid'][key]) for key in gpr1_performance['t_valid']}

		# gpr2_t_valid_mins = {key: np.min(gpr2_performance['t_valid'][key]) for key in gpr2_performance['t_valid']}
		# gpr2_t_valid_maxes = {key: np.max(gpr2_performance['t_valid'][key]) for key in gpr2_performance['t_valid']}
		# gpr2_t_valid_medians = {key: np.median(gpr2_performance['t_valid'][key]) for key in gpr2_performance['t_valid']}
		# gpr2_t_valid_means = {key: np.mean(gpr2_performance['t_valid'][key]) for key in gpr2_performance['t_valid']}
		# gpr2_t_valid_stds = {key: np.std(gpr2_performance['t_valid'][key]) for key in gpr2_performance['t_valid']}

		t_valid_medians = {}
		t_valid_stds = {}
		rnn_t_valid_medians = {}
		rnn_t_valid_stds = {}
		for is_resid in [True,False]:
			# t_valid_mins = {key: np.min(hybrid_performance['t_valid'][key]) for key in hybrid_performance['t_valid']}
			# t_valid_maxes = {key: np.max(hybrid_performance['t_valid'][key]) for key in hybrid_performance['t_valid']}
			# t_valid_means = {key: np.mean(hybrid_performance['t_valid'][key]) for key in hybrid_performance['t_valid']}
			t_valid_medians[is_resid] = {key: np.median(hybrid_performance[is_resid]['t_valid'][key]) for key in hybrid_performance[is_resid]['t_valid']}
			t_valid_stds[is_resid] = {key: np.std(hybrid_performance[is_resid]['t_valid'][key]) for key in hybrid_performance[is_resid]['t_valid']}

			# rnn_t_valid_mins[is_resid] = np.min(rnn_performance[is_resid]['t_valid'])
			# rnn_t_valid_maxes[is_resid] = np.max(rnn_performance[is_resid]['t_valid'])
			# rnn_t_valid_means[is_resid] = np.mean(rnn_performance[is_resid]['t_valid'])
			rnn_t_valid_medians[is_resid] = np.median(rnn_performance[is_resid]['t_valid'])
			rnn_t_valid_stds[is_resid] = np.std(rnn_performance[is_resid]['t_valid'])

		# plotting
		for gp in range(n_gprs):
			if gp==3:
				gp_nm = 'vanilla GPR'
			else:
				gp_nm = 'hybrid GPR {0}'.format(gp+1)
			# hybrid GPR
			eps_vec = sorted(gpr_t_valid_medians[gp].keys())
			median_vec = [gpr_t_valid_medians[gp][eps] for eps in eps_vec]
			std_vec = [gpr_t_valid_stds[gp][eps] for eps in eps_vec]
			ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label=gp_nm, linestyle='--')

		try:
			special_eps_vec = [foo for foo in eps_vec if foo!=0]
			median_vec = [ode_t_valid_medians[eps] for eps in special_eps_vec]
			std_vec = [ode_t_valid_stds[eps] for eps in special_eps_vec]
			ax2.errorbar(x=special_eps_vec, y=median_vec, yerr=std_vec, label='model-only', color='red')
		except:
			pass


		# # hybrid GPR 1
		# eps_vec = sorted(gpr1_t_valid_medians.keys())
		# median_vec = [gpr1_t_valid_medians[eps] for eps in eps_vec]
		# std_vec = [gpr1_t_valid_stds[eps] for eps in eps_vec]
		# ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid GPR 1', color='green', linestyle=':')

		# # hybrid GPR 2
		# eps_vec = sorted(gpr2_t_valid_medians.keys())
		# median_vec = [gpr2_t_valid_medians[eps] for eps in eps_vec]
		# std_vec = [gpr2_t_valid_stds[eps] for eps in eps_vec]
		# ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid GPR 2', color='green', linestyle='--')

		for is_resid in [True,False]:
			# hybrid RNN
			eps_vec = sorted(t_valid_medians[is_resid].keys())
			median_vec = [t_valid_medians[is_resid][eps] for eps in eps_vec]
			std_vec = [t_valid_stds[is_resid][eps] for eps in eps_vec]
			ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid RNN'+is_resid*' residual', color='blue')

			try:
				ax2.errorbar(x=eps_vec, y=[rnn_t_valid_medians[is_resid]]*len(eps_vec), yerr=[rnn_t_valid_stds[is_resid]]*len(eps_vec) ,label='vanilla RNN'+is_resid*' residual', color='black')
			except:
				pass

		ax2.set_xlabel('epsilon Model Error')
		ax2.set_ylabel('Validity Time')
		ax2.legend()

	fig.suptitle('Performance on Test Set Under Varying Model Error')
	fig.savefig(fname=output_fname)

	ax1.set_xscale('log')
	ax2.set_xscale('log')
	fig.savefig(fname=output_fname + '_xlog')
	plt.close(fig)

	try:
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
					# pdb.set_trace()
					pass

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

		ax2.set_xscale('log')
		fig.savefig(fname=output_fname+'_train_time_xlog')
		plt.close(fig)
	except:
		print('Couldnt plot train time...probably because didnt set it up to recognize residual methods yet')
		pass


def extract_performance1(my_dirs, output_fname="./performance_comparisons", win=1, many_epochs=True, my_token='hs', n_gprs=0, my_labels={'xlabel': 'Experimental Variable'}):
	# first, get sizes of things...max window size is 10% of whole test set.
	init_dirs = [x for x in my_dirs if 'RNN' in x.split('/')[-1]]
	# d_label = my_dirs[0].split("/")[-1].rstrip('_noisy').rstrip('_clean')
	x_test = pd.DataFrame(np.loadtxt(init_dirs[0]+"/loss_vec_clean_test.txt",ndmin=2))
	n_vals = x_test.shape[0]

	win = min(win,n_vals//3)
	model_performance = {'mse':(), 't_valid':()}
	rnn_performance = {'mse':{}, 'mse_time':{}, 't_valid':{}, 't_valid_time':{}}
	hybrid_performance = {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}}
	dict_performance = {'rnn':rnn_performance,
						'mechRNN': {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}},
						# 'hybrid GPR1': {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}},
						# 'hybrid GPR2': {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}},
						 'model': model_performance}
	for gp in range(n_gprs):
		gp_nm = 'hybrid GPR{0}'.format(gp+1)
		dict_performance[gp_nm] = {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}}


	for d in my_dirs:
		d_label = d.split("/")[-1].rstrip('_noisy').rstrip('_clean')
		my_feature = float([z.strip(my_token) for z in d_label.split('_') if my_token in z][-1])
		try:
			if ('vanilla' not in d_label) and ('GPR' not in d_label):
				model_loss = np.loadtxt(d+'/perfectModel_loss_vec_clean_test.txt',ndmin=1)
				model_t_valid = np.loadtxt(d+'/perfectModel_prediction_validity_time_clean_test.txt',ndmin=1)
				for kkt in range(model_loss.shape[0]):
					dict_performance['model']['mse'] += (float(model_loss[kkt]),)
					dict_performance['model']['t_valid'] += (float(model_t_valid[kkt]),)
		except:
			pdb.set_trace()
			pass

		# x_train = pd.DataFrame(np.loadtxt(d+"/loss_vec_train.txt"))
		try:
			x_test = pd.DataFrame(np.loadtxt(d+"/loss_vec_clean_test.txt",ndmin=2))
		except:
			pdb.set_trace()

		if many_epochs:
			# try:
			x_valid_test = pd.DataFrame(np.loadtxt(d+"/prediction_validity_time_clean_test.txt",ndmin=2))
			# except:
			# 	x_valid_test = np.loadtxt(d+"/prediction_validity_time_clean_test.txt")
			# x_kl_test = pd.DataFrame(np.loadtxt(d+"/kl_vec_inv_clean_test.txt"))
		if win and 'GPR' not in d_label:
			# x_train = x_train.rolling(win).mean()
			x_test = x_test.rolling(win).mean()
			if many_epochs:
				x_valid_test = x_valid_test.rolling(win).mean()
				# for kk in plot_state_indices:
				# 	ax4.plot(x_kl_test.loc[:,kk].rolling(win).mean(), label=d_label)
		# if my_feature is None:
		# 	rnn_performance['mse'] += (float(np.min(x_test)),)
		# 	if many_epochs:
		# 		rnn_performance['t_valid'] += (float(np.max(x_valid_test)),)
		if 'vanilla' in d_label:
			mtype = 'rnn'
		elif 'mech' in d_label:
			mtype = 'mechRNN'
		elif 'GPR' in d_label:
			mtype = 'hybrid GPR{0}'.format(d_label[d_label.find('GPR') + 3])
		else:
			pdb.set_trace()

		n_tests = x_valid_test.shape[1]
		for kkt in range(n_tests):
			if my_feature in dict_performance[mtype]['mse']:
				dict_performance[mtype]['mse'][my_feature] += (float(np.min(x_test.loc[:,kkt])),)
				dict_performance[mtype]['mse_time'][my_feature] += (float(np.nanargmin(x_test.loc[:,kkt])),)
				if many_epochs:
					dict_performance[mtype]['t_valid'][my_feature] += (float(np.max(x_valid_test.loc[:,kkt])),)
					dict_performance[mtype]['t_valid_time'][my_feature] += (float(np.nanargmax(x_valid_test.loc[:,kkt])),)
			else:
				dict_performance[mtype]['mse'][my_feature] = (float(np.min(x_test.loc[:,kkt])),)
				dict_performance[mtype]['mse_time'][my_feature] = (float(np.nanargmin(x_test.loc[:,kkt])),)
				if many_epochs:
					dict_performance[mtype]['t_valid'][my_feature] = (float(np.max(x_valid_test.loc[:,kkt])),)
					dict_performance[mtype]['t_valid_time'][my_feature] = (float(np.nanargmax(x_valid_test.loc[:,kkt])),)

	# now summarize
	test_loss_mins = {key: np.min(dict_performance['mechRNN']['mse'][key]) for key in dict_performance['mechRNN']['mse']}
	test_loss_maxes = {key: np.max(dict_performance['mechRNN']['mse'][key]) for key in dict_performance['mechRNN']['mse']}
	test_loss_means = {key: np.mean(dict_performance['mechRNN']['mse'][key]) for key in dict_performance['mechRNN']['mse']}
	test_loss_medians = {key: np.median(dict_performance['mechRNN']['mse'][key]) for key in dict_performance['mechRNN']['mse']}
	test_loss_stds = {key: np.std(dict_performance['mechRNN']['mse'][key]) for key in dict_performance['mechRNN']['mse']}


	gpr_test_loss_mins = {}
	gpr_test_loss_maxes = {}
	gpr_test_loss_means = {}
	gpr_test_loss_medians = {}
	gpr_test_loss_stds = {}
	for gp in range(n_gprs):
		gp_nm = 'hybrid GPR{0}'.format(gp+1)
		gpr_test_loss_mins[gp_nm] = {key: np.min(dict_performance[gp_nm]['mse'][key]) for key in dict_performance[gp_nm]['mse']}
		gpr_test_loss_maxes[gp_nm] = {key: np.max(dict_performance[gp_nm]['mse'][key]) for key in dict_performance[gp_nm]['mse']}
		gpr_test_loss_means[gp_nm] = {key: np.mean(dict_performance[gp_nm]['mse'][key]) for key in dict_performance[gp_nm]['mse']}
		gpr_test_loss_medians[gp_nm] = {key: np.median(dict_performance[gp_nm]['mse'][key]) for key in dict_performance[gp_nm]['mse']}
		gpr_test_loss_stds[gp_nm] = {key: np.std(dict_performance[gp_nm]['mse'][key]) for key in dict_performance[gp_nm]['mse']}

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

	# mechRNN
	eps_vec = sorted(test_loss_medians.keys())
	median_vec = [test_loss_medians[eps] for eps in eps_vec]
	std_vec = [test_loss_stds[eps] for eps in eps_vec]
	ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='mechRNN (eps=0.05)', color='blue')

	for gp_nm in gpr_test_loss_medians:
		# Hybrid GPR 1
		eps_vec = sorted(gpr_test_loss_medians[gp_nm].keys())
		median_vec = [gpr_test_loss_medians[gp_nm][eps] for eps in eps_vec]
		std_vec = [gpr_test_loss_stds[gp_nm][eps] for eps in eps_vec]
		ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label=gp_nm + ' (eps=0.05)',linestyle='--')

	# # Hybrid GPR 1
	# eps_vec = sorted(gpr1_test_loss_medians.keys())
	# median_vec = [gpr1_test_loss_medians[eps] for eps in eps_vec]
	# std_vec = [gpr1_test_loss_stds[eps] for eps in eps_vec]
	# ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid GPR 1 (eps=0.05)', color='green', linestyle=':')

	# # Hybrid GPR 1
	# eps_vec = sorted(gpr2_test_loss_medians.keys())
	# median_vec = [gpr2_test_loss_medians[eps] for eps in eps_vec]
	# std_vec = [gpr2_test_loss_stds[eps] for eps in eps_vec]
	# ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid GPR 2 (eps=0.05)', color='green', linestyle='--')


	ax1.set_xlabel(my_labels['xlabel'])
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
		t_valid_mins = {key: np.min(dict_performance['mechRNN']['t_valid'][key]) for key in dict_performance['mechRNN']['t_valid']}
		t_valid_maxes = {key: np.max(dict_performance['mechRNN']['t_valid'][key]) for key in dict_performance['mechRNN']['t_valid']}
		t_valid_means = {key: np.mean(dict_performance['mechRNN']['t_valid'][key]) for key in dict_performance['mechRNN']['t_valid']}
		t_valid_medians = {key: np.median(dict_performance['mechRNN']['t_valid'][key]) for key in dict_performance['mechRNN']['t_valid']}
		t_valid_stds = {key: np.std(dict_performance['mechRNN']['t_valid'][key]) for key in dict_performance['mechRNN']['t_valid']}

		gpr_t_valid_mins = {}
		gpr_t_valid_maxes = {}
		gpr_t_valid_means = {}
		gpr_t_valid_medians = {}
		gpr_t_valid_stds = {}
		for gp in range(n_gprs):
			gp_nm = 'hybrid GPR{0}'.format(gp+1)
			gpr_t_valid_mins[gp_nm] = {key: np.min(dict_performance[gp_nm]['t_valid'][key]) for key in dict_performance[gp_nm]['t_valid']}
			gpr_t_valid_maxes[gp_nm] = {key: np.max(dict_performance[gp_nm]['t_valid'][key]) for key in dict_performance[gp_nm]['t_valid']}
			gpr_t_valid_means[gp_nm] = {key: np.mean(dict_performance[gp_nm]['t_valid'][key]) for key in dict_performance[gp_nm]['t_valid']}
			gpr_t_valid_medians[gp_nm] = {key: np.median(dict_performance[gp_nm]['t_valid'][key]) for key in dict_performance[gp_nm]['t_valid']}
			gpr_t_valid_stds[gp_nm] = {key: np.std(dict_performance[gp_nm]['t_valid'][key]) for key in dict_performance[gp_nm]['t_valid']}

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

		# mechRNN
		eps_vec = sorted(t_valid_medians.keys())
		median_vec = [t_valid_medians[eps] for eps in eps_vec]
		std_vec = [t_valid_stds[eps] for eps in eps_vec]
		ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='mechRNN (eps=0.05)', color='blue')

		for gp_nm in gpr_t_valid_medians:
			# Hybrid GPR 1
			eps_vec = sorted(gpr_t_valid_medians[gp_nm].keys())
			median_vec = [gpr_t_valid_medians[gp_nm][eps] for eps in eps_vec]
			std_vec = [gpr_t_valid_stds[gp_nm][eps] for eps in eps_vec]
			ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label=gp_nm + ' (eps=0.05)',linestyle='--')

		# # Hybrid GPR 1
		# eps_vec = sorted(gpr1_t_valid_medians.keys())
		# median_vec = [gpr1_t_valid_medians[eps] for eps in eps_vec]
		# std_vec = [gpr1_t_valid_stds[eps] for eps in eps_vec]
		# ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid GPR 1 (eps=0.05)', color='green', linestyle=':')

		# # Hybrid GPR 2
		# eps_vec = sorted(gpr2_t_valid_medians.keys())
		# median_vec = [gpr2_t_valid_medians[eps] for eps in eps_vec]
		# std_vec = [gpr2_t_valid_stds[eps] for eps in eps_vec]
		# ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid GPR 2 (eps=0.05)', color='green', linestyle='--')

		ax2.set_xlabel(my_labels['xlabel'])
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

	fig.suptitle(r'Performance on Test Set Under Varying training data quantity')
	fig.savefig(fname=output_fname)

	ax1.set_xscale('log')
	ax2.set_xscale('log')
	fig.savefig(fname=output_fname + '_xlog')
	plt.close(fig)

	# plot summary of Time to Train
	test_loss_mins = {key: np.min(dict_performance['mechRNN']['mse_time'][key]) for key in dict_performance['mechRNN']['mse_time']}
	test_loss_maxes = {key: np.max(dict_performance['mechRNN']['mse_time'][key]) for key in dict_performance['mechRNN']['mse_time']}
	test_loss_means = {key: np.mean(dict_performance['mechRNN']['mse_time'][key]) for key in dict_performance['mechRNN']['mse_time']}
	test_loss_medians = {key: np.median(dict_performance['mechRNN']['mse_time'][key]) for key in dict_performance['mechRNN']['mse_time']}
	test_loss_stds = {key: np.std(dict_performance['mechRNN']['mse_time'][key]) for key in dict_performance['mechRNN']['mse_time']}
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

	ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='mechRNN (eps=0.05)', color='blue')
	ax1.set_xlabel(my_labels['xlabel'])
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
		t_valid_mins = {key: np.min(dict_performance['mechRNN']['t_valid_time'][key]) for key in dict_performance['mechRNN']['t_valid_time']}
		t_valid_maxes = {key: np.max(dict_performance['mechRNN']['t_valid_time'][key]) for key in dict_performance['mechRNN']['t_valid_time']}
		t_valid_means = {key: np.mean(dict_performance['mechRNN']['t_valid_time'][key]) for key in dict_performance['mechRNN']['t_valid_time']}
		t_valid_medians = {key: np.median(dict_performance['mechRNN']['t_valid_time'][key]) for key in dict_performance['mechRNN']['t_valid_time']}
		t_valid_stds = {key: np.std(dict_performance['mechRNN']['t_valid_time'][key]) for key in dict_performance['mechRNN']['t_valid_time']}

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

		ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='mechRNN (eps=0.05)', color='blue')
		ax2.set_xlabel(my_labels['xlabel'])
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

	fig.suptitle('Train-time until reaching optimal test performance')
	fig.savefig(fname=output_fname+'_train_time')
	plt.close(fig)


def extract_hidden_size_performance(my_dirs, output_fname="./hidden_size_comparisons", win=1000, many_epochs=True, hs_token='hs'):

	# first, get sizes of things...max window size is 10% of whole test set.
	init_dirs = [x for x in my_dirs if 'RNN' in x.split('/')[-1]]
	x_test = pd.DataFrame(np.loadtxt(init_dirs[0]+"/loss_vec_clean_test.txt"))
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
			model_loss = np.loadtxt(d+'/perfectModel_loss_clean_test.txt')
			model_t_valid = np.loadtxt(d+'/perfectModel_validity_time_clean_test.txt')
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



def extract_n_data_performance(my_dirs, output_fname="./n_data_comparisons", win=1, many_epochs=True, dt_token='Ndata_'):
	n_gprs = 4
	# first, get sizes of things...max window size is 10% of whole test set.
	init_dirs = [x for x in my_dirs if 'RNN' in x.split('/')[-1]]
	# d_label = my_dirs[0].split("/")[-1].rstrip('_noisy').rstrip('_clean')
	x_test = pd.DataFrame(np.loadtxt(init_dirs[0]+"/loss_vec_clean_test.txt",ndmin=2))
	n_vals = x_test.shape[0]

	win = min(win,n_vals//3)
	model_performance = {'mse':(), 't_valid':()}
	rnn_performance = {'mse':{}, 'mse_time':{}, 't_valid':{}, 't_valid_time':{}}
	hybrid_performance = {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}}
	dict_performance = {'rnn':rnn_performance,
						'mechRNN': {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}},
						# 'hybrid GPR1': {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}},
						# 'hybrid GPR2': {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}},
						 'model': model_performance}
	for gp in range(n_gprs):
		gp_nm = 'hybrid GPR{0}'.format(gp+1)
		dict_performance[gp_nm] = {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}}


	for d in my_dirs:
		d_label = d.split("/")[-1].rstrip('_noisy').rstrip('_clean')
		my_dt = float(d.split("/")[-3].lstrip(dt_token))
		try:
			if ('vanilla' not in d_label) and ('GPR' not in d_label):
				model_loss = np.loadtxt(d+'/perfectModel_loss_vec_clean_test.txt',ndmin=1)
				model_t_valid = np.loadtxt(d+'/perfectModel_prediction_validity_time_clean_test.txt',ndmin=1)
				for kkt in range(model_loss.shape[0]):
					dict_performance['model']['mse'] += (float(model_loss[kkt]),)
					dict_performance['model']['t_valid'] += (float(model_t_valid[kkt]),)
		except:
			pdb.set_trace()
			pass

		# x_train = pd.DataFrame(np.loadtxt(d+"/loss_vec_train.txt"))
		try:
			x_test = pd.DataFrame(np.loadtxt(d+"/loss_vec_clean_test.txt",ndmin=2))
		except:
			pdb.set_trace()

		if many_epochs:
			# try:
			x_valid_test = pd.DataFrame(np.loadtxt(d+"/prediction_validity_time_clean_test.txt",ndmin=2))
			# except:
			# 	x_valid_test = np.loadtxt(d+"/prediction_validity_time_clean_test.txt")
			# x_kl_test = pd.DataFrame(np.loadtxt(d+"/kl_vec_inv_clean_test.txt"))
		if win and 'GPR' not in d_label:
			# x_train = x_train.rolling(win).mean()
			x_test = x_test.rolling(win).mean()
			if many_epochs:
				x_valid_test = x_valid_test.rolling(win).mean()
				# for kk in plot_state_indices:
				# 	ax4.plot(x_kl_test.loc[:,kk].rolling(win).mean(), label=d_label)
		# if my_dt is None:
		# 	rnn_performance['mse'] += (float(np.min(x_test)),)
		# 	if many_epochs:
		# 		rnn_performance['t_valid'] += (float(np.max(x_valid_test)),)
		if 'vanilla' in d_label:
			mtype = 'rnn'
		elif 'mech' in d_label:
			mtype = 'mechRNN'
		elif 'GPR' in d_label:
			mtype = 'hybrid GPR{0}'.format(d_label[d_label.find('GPR') + 3])
		else:
			pdb.set_trace()

		n_tests = x_valid_test.shape[1]
		for kkt in range(n_tests):
			if my_dt in dict_performance[mtype]['mse']:
				dict_performance[mtype]['mse'][my_dt] += (float(np.min(x_test.loc[:,kkt])),)
				dict_performance[mtype]['mse_time'][my_dt] += (float(np.nanargmin(x_test.loc[:,kkt])),)
				if many_epochs:
					dict_performance[mtype]['t_valid'][my_dt] += (float(np.max(x_valid_test.loc[:,kkt])),)
					dict_performance[mtype]['t_valid_time'][my_dt] += (float(np.nanargmax(x_valid_test.loc[:,kkt])),)
			else:
				dict_performance[mtype]['mse'][my_dt] = (float(np.min(x_test.loc[:,kkt])),)
				dict_performance[mtype]['mse_time'][my_dt] = (float(np.nanargmin(x_test.loc[:,kkt])),)
				if many_epochs:
					dict_performance[mtype]['t_valid'][my_dt] = (float(np.max(x_valid_test.loc[:,kkt])),)
					dict_performance[mtype]['t_valid_time'][my_dt] = (float(np.nanargmax(x_valid_test.loc[:,kkt])),)

	# now summarize
	test_loss_mins = {key: np.min(dict_performance['mechRNN']['mse'][key]) for key in dict_performance['mechRNN']['mse']}
	test_loss_maxes = {key: np.max(dict_performance['mechRNN']['mse'][key]) for key in dict_performance['mechRNN']['mse']}
	test_loss_means = {key: np.mean(dict_performance['mechRNN']['mse'][key]) for key in dict_performance['mechRNN']['mse']}
	test_loss_medians = {key: np.median(dict_performance['mechRNN']['mse'][key]) for key in dict_performance['mechRNN']['mse']}
	test_loss_stds = {key: np.std(dict_performance['mechRNN']['mse'][key]) for key in dict_performance['mechRNN']['mse']}


	gpr_test_loss_mins = {}
	gpr_test_loss_maxes = {}
	gpr_test_loss_means = {}
	gpr_test_loss_medians = {}
	gpr_test_loss_stds = {}
	for gp in range(n_gprs):
		gp_nm = 'hybrid GPR{0}'.format(gp+1)
		gpr_test_loss_mins[gp_nm] = {key: np.min(dict_performance[gp_nm]['mse'][key]) for key in dict_performance[gp_nm]['mse']}
		gpr_test_loss_maxes[gp_nm] = {key: np.max(dict_performance[gp_nm]['mse'][key]) for key in dict_performance[gp_nm]['mse']}
		gpr_test_loss_means[gp_nm] = {key: np.mean(dict_performance[gp_nm]['mse'][key]) for key in dict_performance[gp_nm]['mse']}
		gpr_test_loss_medians[gp_nm] = {key: np.median(dict_performance[gp_nm]['mse'][key]) for key in dict_performance[gp_nm]['mse']}
		gpr_test_loss_stds[gp_nm] = {key: np.std(dict_performance[gp_nm]['mse'][key]) for key in dict_performance[gp_nm]['mse']}

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

	# mechRNN
	eps_vec = sorted(test_loss_medians.keys())
	median_vec = [test_loss_medians[eps] for eps in eps_vec]
	std_vec = [test_loss_stds[eps] for eps in eps_vec]
	ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='mechRNN (eps=0.05)', color='blue')

	for gp_nm in gpr_test_loss_medians:
		# Hybrid GPR 1
		eps_vec = sorted(gpr_test_loss_medians[gp_nm].keys())
		median_vec = [gpr_test_loss_medians[gp_nm][eps] for eps in eps_vec]
		std_vec = [gpr_test_loss_stds[gp_nm][eps] for eps in eps_vec]
		ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label=gp_nm + ' (eps=0.05)',linestyle='--')

	# # Hybrid GPR 1
	# eps_vec = sorted(gpr1_test_loss_medians.keys())
	# median_vec = [gpr1_test_loss_medians[eps] for eps in eps_vec]
	# std_vec = [gpr1_test_loss_stds[eps] for eps in eps_vec]
	# ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid GPR 1 (eps=0.05)', color='green', linestyle=':')

	# # Hybrid GPR 1
	# eps_vec = sorted(gpr2_test_loss_medians.keys())
	# median_vec = [gpr2_test_loss_medians[eps] for eps in eps_vec]
	# std_vec = [gpr2_test_loss_stds[eps] for eps in eps_vec]
	# ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid GPR 2 (eps=0.05)', color='green', linestyle='--')


	ax1.set_xlabel('Number of Training Points')
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
		t_valid_mins = {key: np.min(dict_performance['mechRNN']['t_valid'][key]) for key in dict_performance['mechRNN']['t_valid']}
		t_valid_maxes = {key: np.max(dict_performance['mechRNN']['t_valid'][key]) for key in dict_performance['mechRNN']['t_valid']}
		t_valid_means = {key: np.mean(dict_performance['mechRNN']['t_valid'][key]) for key in dict_performance['mechRNN']['t_valid']}
		t_valid_medians = {key: np.median(dict_performance['mechRNN']['t_valid'][key]) for key in dict_performance['mechRNN']['t_valid']}
		t_valid_stds = {key: np.std(dict_performance['mechRNN']['t_valid'][key]) for key in dict_performance['mechRNN']['t_valid']}

		gpr_t_valid_mins = {}
		gpr_t_valid_maxes = {}
		gpr_t_valid_means = {}
		gpr_t_valid_medians = {}
		gpr_t_valid_stds = {}
		for gp in range(n_gprs):
			gp_nm = 'hybrid GPR{0}'.format(gp+1)
			gpr_t_valid_mins[gp_nm] = {key: np.min(dict_performance[gp_nm]['t_valid'][key]) for key in dict_performance[gp_nm]['t_valid']}
			gpr_t_valid_maxes[gp_nm] = {key: np.max(dict_performance[gp_nm]['t_valid'][key]) for key in dict_performance[gp_nm]['t_valid']}
			gpr_t_valid_means[gp_nm] = {key: np.mean(dict_performance[gp_nm]['t_valid'][key]) for key in dict_performance[gp_nm]['t_valid']}
			gpr_t_valid_medians[gp_nm] = {key: np.median(dict_performance[gp_nm]['t_valid'][key]) for key in dict_performance[gp_nm]['t_valid']}
			gpr_t_valid_stds[gp_nm] = {key: np.std(dict_performance[gp_nm]['t_valid'][key]) for key in dict_performance[gp_nm]['t_valid']}

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

		# mechRNN
		eps_vec = sorted(t_valid_medians.keys())
		median_vec = [t_valid_medians[eps] for eps in eps_vec]
		std_vec = [t_valid_stds[eps] for eps in eps_vec]
		ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='mechRNN (eps=0.05)', color='blue')

		for gp_nm in gpr_t_valid_medians:
			# Hybrid GPR 1
			eps_vec = sorted(gpr_t_valid_medians[gp_nm].keys())
			median_vec = [gpr_t_valid_medians[gp_nm][eps] for eps in eps_vec]
			std_vec = [gpr_t_valid_stds[gp_nm][eps] for eps in eps_vec]
			ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label=gp_nm + ' (eps=0.05)',linestyle='--')

		# # Hybrid GPR 1
		# eps_vec = sorted(gpr1_t_valid_medians.keys())
		# median_vec = [gpr1_t_valid_medians[eps] for eps in eps_vec]
		# std_vec = [gpr1_t_valid_stds[eps] for eps in eps_vec]
		# ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid GPR 1 (eps=0.05)', color='green', linestyle=':')

		# # Hybrid GPR 2
		# eps_vec = sorted(gpr2_t_valid_medians.keys())
		# median_vec = [gpr2_t_valid_medians[eps] for eps in eps_vec]
		# std_vec = [gpr2_t_valid_stds[eps] for eps in eps_vec]
		# ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid GPR 2 (eps=0.05)', color='green', linestyle='--')

		ax2.set_xlabel('Number of Training Points')
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

	fig.suptitle(r'Performance on Test Set Under Varying training data quantity')
	fig.savefig(fname=output_fname)

	ax1.set_xscale('log')
	ax2.set_xscale('log')
	fig.savefig(fname=output_fname + '_xlog')
	plt.close(fig)

	# plot summary of Time to Train
	test_loss_mins = {key: np.min(dict_performance['mechRNN']['mse_time'][key]) for key in dict_performance['mechRNN']['mse_time']}
	test_loss_maxes = {key: np.max(dict_performance['mechRNN']['mse_time'][key]) for key in dict_performance['mechRNN']['mse_time']}
	test_loss_means = {key: np.mean(dict_performance['mechRNN']['mse_time'][key]) for key in dict_performance['mechRNN']['mse_time']}
	test_loss_medians = {key: np.median(dict_performance['mechRNN']['mse_time'][key]) for key in dict_performance['mechRNN']['mse_time']}
	test_loss_stds = {key: np.std(dict_performance['mechRNN']['mse_time'][key]) for key in dict_performance['mechRNN']['mse_time']}
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

	ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='mechRNN (eps=0.05)', color='blue')
	ax1.set_xlabel('Number of Training Points')
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
		t_valid_mins = {key: np.min(dict_performance['mechRNN']['t_valid_time'][key]) for key in dict_performance['mechRNN']['t_valid_time']}
		t_valid_maxes = {key: np.max(dict_performance['mechRNN']['t_valid_time'][key]) for key in dict_performance['mechRNN']['t_valid_time']}
		t_valid_means = {key: np.mean(dict_performance['mechRNN']['t_valid_time'][key]) for key in dict_performance['mechRNN']['t_valid_time']}
		t_valid_medians = {key: np.median(dict_performance['mechRNN']['t_valid_time'][key]) for key in dict_performance['mechRNN']['t_valid_time']}
		t_valid_stds = {key: np.std(dict_performance['mechRNN']['t_valid_time'][key]) for key in dict_performance['mechRNN']['t_valid_time']}

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

		ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='mechRNN (eps=0.05)', color='blue')
		ax2.set_xlabel('Number of Training Points')
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

	fig.suptitle(r'Train-time until reaching optimal test performance (w.r.t. time step $\Delta t$)')
	fig.savefig(fname=output_fname+'_train_time')
	plt.close(fig)


def extract_delta_t_performance(my_dirs, output_fname="./delta_t_comparisons", win=1, many_epochs=True, dt_token='dt_'):
	n_gprs = 4
	# first, get sizes of things...max window size is 10% of whole test set.
	init_dirs = [x for x in my_dirs if 'RNN' in x.split('/')[-1]]
	# d_label = my_dirs[0].split("/")[-1].rstrip('_noisy').rstrip('_clean')
	x_test = pd.DataFrame(np.loadtxt(init_dirs[0]+"/loss_vec_clean_test.txt",ndmin=2))
	n_vals = x_test.shape[0]

	win = min(win,n_vals//3)
	model_performance = {'mse':(), 't_valid':()}
	rnn_performance = {'mse':{}, 'mse_time':{}, 't_valid':{}, 't_valid_time':{}}
	hybrid_performance = {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}}
	dict_performance = {'rnn':rnn_performance,
						'mechRNN': {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}},
						# 'hybrid GPR1': {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}},
						# 'hybrid GPR2': {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}},
						 'model': model_performance}
	for gp in range(n_gprs):
		gp_nm = 'hybrid GPR{0}'.format(gp+1)
		dict_performance[gp_nm] = {'mse':{}, 't_valid':{}, 'mse_time':{}, 't_valid_time':{}}


	for d in my_dirs:
		d_label = d.split("/")[-1].rstrip('_noisy').rstrip('_clean')
		my_dt = float(d.split("/")[-3].lstrip(dt_token))
		try:
			if ('vanilla' not in d_label) and ('GPR' not in d_label):
				model_loss = np.loadtxt(d+'/perfectModel_loss_vec_clean_test.txt',ndmin=1)
				model_t_valid = np.loadtxt(d+'/perfectModel_prediction_validity_time_clean_test.txt',ndmin=1)
				for kkt in range(model_loss.shape[0]):
					dict_performance['model']['mse'] += (float(model_loss[kkt]),)
					dict_performance['model']['t_valid'] += (float(model_t_valid[kkt]),)
		except:
			pdb.set_trace()
			pass

		# x_train = pd.DataFrame(np.loadtxt(d+"/loss_vec_train.txt"))
		try:
			x_test = pd.DataFrame(np.loadtxt(d+"/loss_vec_clean_test.txt",ndmin=2))
		except:
			pdb.set_trace()

		if many_epochs:
			# try:
			x_valid_test = pd.DataFrame(np.loadtxt(d+"/prediction_validity_time_clean_test.txt",ndmin=2))
			# except:
			# 	x_valid_test = np.loadtxt(d+"/prediction_validity_time_clean_test.txt")
			# x_kl_test = pd.DataFrame(np.loadtxt(d+"/kl_vec_inv_clean_test.txt"))
		if win and 'GPR' not in d_label:
			# x_train = x_train.rolling(win).mean()
			x_test = x_test.rolling(win).mean()
			if many_epochs:
				x_valid_test = x_valid_test.rolling(win).mean()
				# for kk in plot_state_indices:
				# 	ax4.plot(x_kl_test.loc[:,kk].rolling(win).mean(), label=d_label)
		# if my_dt is None:
		# 	rnn_performance['mse'] += (float(np.min(x_test)),)
		# 	if many_epochs:
		# 		rnn_performance['t_valid'] += (float(np.max(x_valid_test)),)
		if 'vanilla' in d_label:
			mtype = 'rnn'
		elif 'mech' in d_label:
			mtype = 'mechRNN'
		elif 'GPR' in d_label:
			mtype = 'hybrid GPR{0}'.format(d_label[d_label.find('GPR') + 3])
		else:
			pdb.set_trace()

		n_tests = x_valid_test.shape[1]
		for kkt in range(n_tests):
			if my_dt in dict_performance[mtype]['mse']:
				dict_performance[mtype]['mse'][my_dt] += (float(np.min(x_test.loc[:,kkt])),)
				dict_performance[mtype]['mse_time'][my_dt] += (float(np.nanargmin(x_test.loc[:,kkt])),)
				if many_epochs:
					dict_performance[mtype]['t_valid'][my_dt] += (float(np.max(x_valid_test.loc[:,kkt])),)
					dict_performance[mtype]['t_valid_time'][my_dt] += (float(np.nanargmax(x_valid_test.loc[:,kkt])),)
			else:
				dict_performance[mtype]['mse'][my_dt] = (float(np.min(x_test.loc[:,kkt])),)
				dict_performance[mtype]['mse_time'][my_dt] = (float(np.nanargmin(x_test.loc[:,kkt])),)
				if many_epochs:
					dict_performance[mtype]['t_valid'][my_dt] = (float(np.max(x_valid_test.loc[:,kkt])),)
					dict_performance[mtype]['t_valid_time'][my_dt] = (float(np.nanargmax(x_valid_test.loc[:,kkt])),)

	# now summarize
	test_loss_mins = {key: np.min(dict_performance['mechRNN']['mse'][key]) for key in dict_performance['mechRNN']['mse']}
	test_loss_maxes = {key: np.max(dict_performance['mechRNN']['mse'][key]) for key in dict_performance['mechRNN']['mse']}
	test_loss_means = {key: np.mean(dict_performance['mechRNN']['mse'][key]) for key in dict_performance['mechRNN']['mse']}
	test_loss_medians = {key: np.median(dict_performance['mechRNN']['mse'][key]) for key in dict_performance['mechRNN']['mse']}
	test_loss_stds = {key: np.std(dict_performance['mechRNN']['mse'][key]) for key in dict_performance['mechRNN']['mse']}


	gpr_test_loss_mins = {}
	gpr_test_loss_maxes = {}
	gpr_test_loss_means = {}
	gpr_test_loss_medians = {}
	gpr_test_loss_stds = {}
	for gp in range(n_gprs):
		gp_nm = 'hybrid GPR{0}'.format(gp+1)
		gpr_test_loss_mins[gp_nm] = {key: np.min(dict_performance[gp_nm]['mse'][key]) for key in dict_performance[gp_nm]['mse']}
		gpr_test_loss_maxes[gp_nm] = {key: np.max(dict_performance[gp_nm]['mse'][key]) for key in dict_performance[gp_nm]['mse']}
		gpr_test_loss_means[gp_nm] = {key: np.mean(dict_performance[gp_nm]['mse'][key]) for key in dict_performance[gp_nm]['mse']}
		gpr_test_loss_medians[gp_nm] = {key: np.median(dict_performance[gp_nm]['mse'][key]) for key in dict_performance[gp_nm]['mse']}
		gpr_test_loss_stds[gp_nm] = {key: np.std(dict_performance[gp_nm]['mse'][key]) for key in dict_performance[gp_nm]['mse']}

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

	# mechRNN
	eps_vec = sorted(test_loss_medians.keys())
	median_vec = [test_loss_medians[eps] for eps in eps_vec]
	std_vec = [test_loss_stds[eps] for eps in eps_vec]
	ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='mechRNN (eps=0.05)', color='blue')

	for gp_nm in gpr_test_loss_medians:
		# Hybrid GPR 1
		eps_vec = sorted(gpr_test_loss_medians[gp_nm].keys())
		median_vec = [gpr_test_loss_medians[gp_nm][eps] for eps in eps_vec]
		std_vec = [gpr_test_loss_stds[gp_nm][eps] for eps in eps_vec]
		ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label=gp_nm + ' (eps=0.05)',linestyle='--')

	# # Hybrid GPR 1
	# eps_vec = sorted(gpr1_test_loss_medians.keys())
	# median_vec = [gpr1_test_loss_medians[eps] for eps in eps_vec]
	# std_vec = [gpr1_test_loss_stds[eps] for eps in eps_vec]
	# ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid GPR 1 (eps=0.05)', color='green', linestyle=':')

	# # Hybrid GPR 1
	# eps_vec = sorted(gpr2_test_loss_medians.keys())
	# median_vec = [gpr2_test_loss_medians[eps] for eps in eps_vec]
	# std_vec = [gpr2_test_loss_stds[eps] for eps in eps_vec]
	# ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid GPR 2 (eps=0.05)', color='green', linestyle='--')

	ax1.set_xlabel(r'$\Delta t$')
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
		t_valid_mins = {key: np.min(dict_performance['mechRNN']['t_valid'][key]) for key in dict_performance['mechRNN']['t_valid']}
		t_valid_maxes = {key: np.max(dict_performance['mechRNN']['t_valid'][key]) for key in dict_performance['mechRNN']['t_valid']}
		t_valid_means = {key: np.mean(dict_performance['mechRNN']['t_valid'][key]) for key in dict_performance['mechRNN']['t_valid']}
		t_valid_medians = {key: np.median(dict_performance['mechRNN']['t_valid'][key]) for key in dict_performance['mechRNN']['t_valid']}
		t_valid_stds = {key: np.std(dict_performance['mechRNN']['t_valid'][key]) for key in dict_performance['mechRNN']['t_valid']}

		gpr_t_valid_mins = {}
		gpr_t_valid_maxes = {}
		gpr_t_valid_means = {}
		gpr_t_valid_medians = {}
		gpr_t_valid_stds = {}
		for gp in range(n_gprs):
			gp_nm = 'hybrid GPR{0}'.format(gp+1)
			gpr_t_valid_mins[gp_nm] = {key: np.min(dict_performance[gp_nm]['t_valid'][key]) for key in dict_performance[gp_nm]['t_valid']}
			gpr_t_valid_maxes[gp_nm] = {key: np.max(dict_performance[gp_nm]['t_valid'][key]) for key in dict_performance[gp_nm]['t_valid']}
			gpr_t_valid_means[gp_nm] = {key: np.mean(dict_performance[gp_nm]['t_valid'][key]) for key in dict_performance[gp_nm]['t_valid']}
			gpr_t_valid_medians[gp_nm] = {key: np.median(dict_performance[gp_nm]['t_valid'][key]) for key in dict_performance[gp_nm]['t_valid']}
			gpr_t_valid_stds[gp_nm] = {key: np.std(dict_performance[gp_nm]['t_valid'][key]) for key in dict_performance[gp_nm]['t_valid']}

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

		# mechRNN
		eps_vec = sorted(t_valid_medians.keys())
		median_vec = [t_valid_medians[eps] for eps in eps_vec]
		std_vec = [t_valid_stds[eps] for eps in eps_vec]
		ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='mechRNN (eps=0.05)', color='blue')

		for gp_nm in gpr_t_valid_medians:
			# Hybrid GPR 1
			eps_vec = sorted(gpr_t_valid_medians[gp_nm].keys())
			median_vec = [gpr_t_valid_medians[gp_nm][eps] for eps in eps_vec]
			std_vec = [gpr_t_valid_stds[gp_nm][eps] for eps in eps_vec]
			ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label=gp_nm + ' (eps=0.05)',linestyle='--')

		# # Hybrid GPR 1
		# eps_vec = sorted(gpr1_t_valid_medians.keys())
		# median_vec = [gpr1_t_valid_medians[eps] for eps in eps_vec]
		# std_vec = [gpr1_t_valid_stds[eps] for eps in eps_vec]
		# ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid GPR 1 (eps=0.05)', color='green', linestyle=':')

		# # Hybrid GPR 2
		# eps_vec = sorted(gpr2_t_valid_medians.keys())
		# median_vec = [gpr2_t_valid_medians[eps] for eps in eps_vec]
		# std_vec = [gpr2_t_valid_stds[eps] for eps in eps_vec]
		# ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='hybrid GPR 2 (eps=0.05)', color='green', linestyle='--')

		ax2.set_xlabel(r'$\Delta t$')
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

	fig.suptitle(r'Performance on Test Set Under Varying training data quantity')
	fig.savefig(fname=output_fname)


	ax1.set_xscale('log')
	ax2.set_xscale('log')
	fig.savefig(fname=output_fname + '_xlog')
	plt.close(fig)

	# plot summary of Time to Train
	test_loss_mins = {key: np.min(dict_performance['mechRNN']['mse_time'][key]) for key in dict_performance['mechRNN']['mse_time']}
	test_loss_maxes = {key: np.max(dict_performance['mechRNN']['mse_time'][key]) for key in dict_performance['mechRNN']['mse_time']}
	test_loss_means = {key: np.mean(dict_performance['mechRNN']['mse_time'][key]) for key in dict_performance['mechRNN']['mse_time']}
	test_loss_medians = {key: np.median(dict_performance['mechRNN']['mse_time'][key]) for key in dict_performance['mechRNN']['mse_time']}
	test_loss_stds = {key: np.std(dict_performance['mechRNN']['mse_time'][key]) for key in dict_performance['mechRNN']['mse_time']}
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

	ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='mechRNN (eps=0.05)', color='blue')
	ax1.set_xlabel(r'$\Delta t$')
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
		t_valid_mins = {key: np.min(dict_performance['mechRNN']['t_valid_time'][key]) for key in dict_performance['mechRNN']['t_valid_time']}
		t_valid_maxes = {key: np.max(dict_performance['mechRNN']['t_valid_time'][key]) for key in dict_performance['mechRNN']['t_valid_time']}
		t_valid_means = {key: np.mean(dict_performance['mechRNN']['t_valid_time'][key]) for key in dict_performance['mechRNN']['t_valid_time']}
		t_valid_medians = {key: np.median(dict_performance['mechRNN']['t_valid_time'][key]) for key in dict_performance['mechRNN']['t_valid_time']}
		t_valid_stds = {key: np.std(dict_performance['mechRNN']['t_valid_time'][key]) for key in dict_performance['mechRNN']['t_valid_time']}

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

		ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label='mechRNN (eps=0.05)', color='blue')
		ax2.set_xlabel(r'$\Delta t$')
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

	fig.suptitle(r'Train-time until reaching optimal test performance (w.r.t. time step $\Delta t$)')
	fig.savefig(fname=output_fname+'_train_time')
	plt.close(fig)