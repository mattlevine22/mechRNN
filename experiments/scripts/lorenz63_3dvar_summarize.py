from utils import *
# from utils import make_RNN_data, get_lorenz_inits, lorenz63
import numpy as np
import torch
import argparse
import pdb
from glob import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='3DVAR')
parser.add_argument('--output_dir', type=str, default='default_output', help='filename for generated data')
parser.add_argument('--n_train_trajectories', type=int, default=1, help='number of training trajectories')
parser.add_argument('--n_test_trajectories', type=int, default=1, help='number of testing trajectories')
FLAGS = parser.parse_args()

MY_DIRS = glob('{0}/*/'.format(FLAGS.output_dir))

def general_summary(my_dirs=None, output_dir='default_output', n_train_trajectories=1, n_test_trajectories=1):

	t_assim = {key: {'assim':{'mean':None, 'median':None, 'std':None, 'all':()}, 'pred':{'mean':None, 'median':None, 'std':None, 'all':()}} for key in my_dirs}
	mse = {key: {'assim':{'mean':None, 'median':None, 'std':None, 'all':()}, 'pred':{'mean':None, 'median':None, 'std':None, 'all':()}} for key in my_dirs}
	for d in my_dirs:
		## store key list of key metrics for each method
		for ntest in range(n_test_trajectories):
			for ntrain in range(n_train_trajectories):
				fname = os.path.join(d, 'Train{0}'.format(ntrain), 'Test{0}'.format(ntest), 'output.npz')
				try:
					npzfile = np.load(fname)
					delta_t = npzfile['model_params'].item().get('delta_t')

					foo_t_index_assim = np.argmax(npzfile['pw_assim_errors'] < npzfile['eps'])
					foo_t_index_pred = np.argmax(npzfile['pw_pred_errors'] < npzfile['eps'])
					t_assim[d]['assim']['all'] += (foo_t_index_assim*delta_t,)
					t_assim[d]['pred']['all'] += (foo_t_index_pred*delta_t,)
					mse[d]['assim']['all'] += (np.mean(npzfile['pw_assim_errors'][foo_t_index_assim:]),)
					mse[d]['pred']['all'] += (np.mean(npzfile['pw_pred_errors'][foo_t_index_pred:]),)
				except:
					# Train/Test combination not available
					pass
		if d not in t_assim:
			continue
		t_assim[d]['assim']['mean'] = np.mean(t_assim[d]['assim']['all'])
		t_assim[d]['assim']['median'] = np.median(t_assim[d]['assim']['all'])
		t_assim[d]['assim']['std'] = np.std(t_assim[d]['assim']['all'])
		t_assim[d]['pred']['mean'] = np.mean(t_assim[d]['pred']['all'])
		t_assim[d]['pred']['median'] = np.median(t_assim[d]['pred']['all'])
		t_assim[d]['pred']['std'] = np.std(t_assim[d]['pred']['all'])

		mse[d]['assim']['mean'] = np.mean(mse[d]['assim']['all'])
		mse[d]['assim']['median'] = np.median(mse[d]['assim']['all'])
		mse[d]['assim']['std'] = np.std(mse[d]['assim']['all'])
		mse[d]['pred']['mean'] = np.mean(mse[d]['pred']['all'])
		mse[d]['pred']['median'] = np.median(mse[d]['pred']['all'])
		mse[d]['pred']['std'] = np.std(mse[d]['pred']['all'])

		## plot summaries of the learning process across training trajectories
		train_vec = []
		G_assim_history = None
		G_assim_history_running_mean = None
		for n in range(n_train_trajectories):
			fname = os.path.join(d, 'Train{0}'.format(n), 'output.npz')
			try:
				npzfile = np.load(fname)
				delta_t = npzfile['model_params'].item().get('delta_t')
				G = npzfile['G_assim_history']
				if G_assim_history is None:
					G_assim_history = np.zeros((n_train_trajectories,G.shape[0],G.shape[1]))
					G_assim_history_running_mean = np.zeros((n_train_trajectories,G.shape[0],G.shape[1]))
				G_assim_history[n,:,:] = npzfile['G_assim_history']
				G_assim_history_running_mean[n,:,:] = npzfile['G_assim_history_running_mean']
				train_vec += [n]
			except:
				# Train not available
				pass

		if len(train_vec)==0:
			continue

		t_plot = np.arange(0,round(G.shape[0]*delta_t,8),delta_t)

		fig, axlist = plt.subplots(nrows=G_assim_history.shape[2], ncols=1, sharex=True)
		for k in range(len(axlist)):
			for n in train_vec:
				axlist[k].plot(t_plot, G_assim_history[n,:,k].squeeze())
			axlist[k].set_ylabel('G_{0}'.format(k))
			# axlist[k].set_yscale('log')
		axlist[k].set_xlabel('time')
		fig.suptitle('3DVAR Training: Assimilation Matrix Sequence')
		fig.savefig(fname=d+'/3DVAR_assimilation_matrix_sequence')
		plt.close(fig)

		fig, axlist = plt.subplots(nrows=G_assim_history_running_mean.shape[2], ncols=1, sharex=True)
		for k in range(len(axlist)):
			for n in train_vec:
				axlist[k].plot(t_plot, G_assim_history_running_mean[n,:,k].squeeze())
			axlist[k].set_ylabel('G_{0}'.format(k))
			# axlist[k].set_yscale('log')
		axlist[k].set_xlabel('time')
		fig.suptitle('3DVAR Training: Assimilation Matrix Convergence (Running Mean)')
		fig.savefig(fname=d+'/3DVAR_assimilation_matrix_runningMean')
		plt.close(fig)


		## plot summaries of Testing performance for a single method
		fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
		for ntest in range(n_test_trajectories):
			for ntrain in train_vec:
				fname = os.path.join(d, 'Train{0}'.format(ntrain), 'Test{0}'.format(ntest), 'output.npz')
				try:
					npzfile = np.load(fname)
					eps = 1
					# eps = npzfile['eps']
					t_plot = np.arange(0,round(npzfile['pw_assim_errors'].shape[0]*delta_t,8),delta_t)
					if npzfile['pw_pred_errors'][0] < 10:
						print('ntest:',ntest, 'ntrain:', ntrain)
					ax0.plot(t_plot, npzfile['pw_pred_errors'], color='blue')
					ax1.plot(t_plot, npzfile['pw_assim_errors'], color='blue')
				except:
					# Train/Test combo not available yet
					pass
		if eps:
			ax0.plot(t_plot, [eps for _ in range(len(t_plot))], color = 'black', linestyle='--', label = r'$\epsilon$')
			ax1.plot(t_plot, [eps for _ in range(len(t_plot))], color = 'black', linestyle='--', label = r'$\epsilon$')

		ax0.set_ylabel('MSE')
		ax0.legend()
		ax0.set_title('1-step Prediction Errors')
		ax1.set_ylabel('MSE')
		ax1.legend()
		ax1.set_title('Assimilation Errors')
		ax1.set_xlabel('time')

		fig.suptitle('3DVAR Testing: Error Convergence')
		fig.savefig(fname=d+'/3DVAR_error_convergence')

		ax0.set_yscale('log')
		ax0.set_ylabel('log MSE')
		ax1.set_yscale('log')
		ax1.set_ylabel('log MSE')
		fig.savefig(fname=d+'/3DVAR_error_convergence_log')
		plt.close(fig)


def epsilon_summary(my_dirs=None, output_dir='default_output', n_train_trajectories=1, n_test_trajectories=1, key_nm='eps'):

	if my_dirs is None:
		my_dirs = glob('{0}/*/'.format(output_dir))

	eps_set = set()
	t_assim = {}
	mse = {}
	G_all = {}
	for d in my_dirs:

		# set up data structures
		method_nm = d.split('_')[-1].strip('/')#.split('+')[0]
		eps_val = float([x for x in d.split('_') if key_nm in x][0].strip(key_nm))
		eps_set.add(eps_val)
		if method_nm not in t_assim:
			t_assim[method_nm] = {}
		if method_nm not in mse:
			mse[method_nm] = {}
		if method_nm not in G_all:
			G_all[method_nm] = {}
		t_assim[method_nm][eps_val] = {'assim':{'mean':None, 'median':None, 'std':None, 'all':()}, 'pred':{'mean':None, 'median':None, 'std':None, 'all':()} }
		mse[method_nm][eps_val] = {'assim':{'mean':None, 'median':None, 'std':None, 'all':()}, 'pred':{'mean':None, 'median':None, 'std':None, 'all':()} }
		G_all[method_nm][eps_val] = {'mean':None, 'median':None, 'std':None}

		# store performance statistics
		for ntest in range(n_test_trajectories):
			for ntrain in range(n_train_trajectories):
				fname = os.path.join(d, 'Train{0}'.format(ntrain), 'Test{0}'.format(ntest), 'output.npz')
				try:
					npzfile = np.load(fname)
					delta_t = npzfile['model_params'].item().get('delta_t')

					foo_t_index_assim = np.argmax(npzfile['pw_assim_errors'] < npzfile['eps'])
					foo_t_index_pred = np.argmax(npzfile['pw_pred_errors'] < npzfile['eps'])
					if foo_t_index_assim==0:
						foo_t_index_assim = np.Inf
					if foo_t_index_pred==0:
						foo_t_index_pred = np.Inf
					t_assim[method_nm][eps_val]['assim']['all'] += (foo_t_index_assim*delta_t,)
					t_assim[method_nm][eps_val]['pred']['all'] += (foo_t_index_pred*delta_t,)
					mse[method_nm][eps_val]['assim']['all'] += (np.mean(npzfile['pw_assim_errors']),)
					mse[method_nm][eps_val]['pred']['all'] += (np.mean(npzfile['pw_pred_errors']),)
				except:
					# Train/Test does not exist
					pass

		G_trained = None
		for ntrain in range(n_train_trajectories):
			fname = os.path.join(d, 'Train{0}'.format(ntrain), 'Test{0}'.format(ntest), 'output.npz')
			try:
				npzfile = np.load(fname)
				G = npzfile['G_assim_history_running_mean']
				if G_trained is None:
					G_trained = np.zeros((n_train_trajectories,G.shape[1]))
				G_trained[ntrain,:] = G[-1,:]
			except:
				# Train/Test does not exist
				pass

		pdb.set_trace()
		G_all[method_nm][eps_val]['mean'] = np.mean(G_trained, axis=0)
		G_all[method_nm][eps_val]['std'] = np.std(G_trained, axis=0)


		if method_nm not in t_assim:
			continue

		t_assim[method_nm][eps_val]['assim']['mean'] = np.mean(t_assim[method_nm][eps_val]['assim']['all'])
		t_assim[method_nm][eps_val]['assim']['median'] = np.median(t_assim[method_nm][eps_val]['assim']['all'])
		t_assim[method_nm][eps_val]['assim']['std'] = np.std(t_assim[method_nm][eps_val]['assim']['all'])
		t_assim[method_nm][eps_val]['pred']['mean'] = np.mean(t_assim[method_nm][eps_val]['pred']['all'])
		t_assim[method_nm][eps_val]['pred']['median'] = np.median(t_assim[method_nm][eps_val]['pred']['all'])
		t_assim[method_nm][eps_val]['pred']['std'] = np.std(t_assim[method_nm][eps_val]['pred']['all'])

		mse[method_nm][eps_val]['assim']['mean'] = np.mean(mse[method_nm][eps_val]['assim']['all'])
		mse[method_nm][eps_val]['assim']['median'] = np.median(mse[method_nm][eps_val]['assim']['all'])
		mse[method_nm][eps_val]['assim']['std'] = np.std(mse[method_nm][eps_val]['assim']['all'])
		mse[method_nm][eps_val]['pred']['mean'] = np.mean(mse[method_nm][eps_val]['pred']['all'])
		mse[method_nm][eps_val]['pred']['median'] = np.median(mse[method_nm][eps_val]['pred']['all'])
		mse[method_nm][eps_val]['pred']['std'] = np.std(mse[method_nm][eps_val]['pred']['all'])


	### NOW initialize plot
	fig, axlist = plt.subplots(nrows=2, ncols=2,
		figsize = [10, 10],
		sharey=False, sharex=False)
	ax0 = axlist[0,0]
	ax1 = axlist[0,1]
	ax2 = axlist[1,0]
	ax3 = axlist[1,1]
	eps_vec = sorted(eps_set)


	for method_nm in t_assim:
		median_vec = [t_assim[method_nm][eps_val]['assim']['median'] for eps_val in eps_vec]
		std_vec = [t_assim[method_nm][eps_val]['assim']['std'] for eps_val in eps_vec]
		ax0.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label=method_nm)

		median_vec = [t_assim[method_nm][eps_val]['pred']['median'] for eps_val in eps_vec]
		std_vec = [t_assim[method_nm][eps_val]['pred']['std'] for eps_val in eps_vec]
		ax1.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label=method_nm)

		median_vec = [mse[method_nm][eps_val]['assim']['median'] for eps_val in eps_vec]
		std_vec = [mse[method_nm][eps_val]['assim']['std'] for eps_val in eps_vec]
		ax2.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label=method_nm)

		median_vec = [mse[method_nm][eps_val]['pred']['median'] for eps_val in eps_vec]
		std_vec = [mse[method_nm][eps_val]['pred']['std'] for eps_val in eps_vec]
		ax3.errorbar(x=eps_vec, y=median_vec, yerr=std_vec, label=method_nm)

	ax1.legend()

	ax2.set_xlabel(r'$\epsilon$ model error')
	ax3.set_xlabel(r'$\epsilon$ model error')

	ax0.set_title('Assimilation Error')
	ax1.set_title('1-step Prediction Error')
	ax0.set_ylabel('t_assim')
	ax1.set_ylabel('t_assim')

	ax2.set_ylabel('MSE')
	ax3.set_ylabel('MSE')

	fig.suptitle('3DVAR Testing Performance')
	fig.savefig(fname=output_dir+'/'+key_nm+'_method_comparison')

	ax2.set_ylabel('logMSE')
	ax3.set_ylabel('logMSE')
	ax2.set_yscale('log')
	ax3.set_yscale('log')

	fig.suptitle('3DVAR Testing Performance')
	fig.savefig(fname=output_dir+'/'+key_nm+'_method_comparison_log')
	plt.close(fig)


	##### Compare methods ######
	method_vec = mse.keys()
	for my_eps in eps_vec:
		Xmse = {}
		Xt = {}
		for m in method_vec:
			if '+' in m:
				h = float(m.split('+')[1].strip('h'))
				lrG = float(m.split('+')[2].strip('lrG'))
				if h not in Xmse:
					Xmse[h] = {}
					Xt[h] = {}
				Xmse[h][lrG] = mse[m][my_eps]['assim']['median']
				Xt[h][lrG] = t_assim[m][my_eps]['assim']['median']

		Ymse = np.array([[Xmse[h][lrG] for lrG in sorted(Xmse[h])] for h in sorted(Xmse)])
		Yt = np.array([[Xt[h][lrG] for lrG in sorted(Xt[h])] for h in sorted(Xt)])

		if len(Ymse)==0:
			continue

		# fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2)
		fig, (ax0) = plt.subplots(nrows=1, ncols=1)

		pdb.set_trace()
		vmin = min(np.min(Ymse),mse['standardAssimilation'][my_eps]['assim']['median'])
		vmax = max(np.max(Ymse),mse['standardAssimilation'][my_eps]['assim']['median'])
		im = ax0.imshow(Ymse, interpolation='none', vmin=vmin, vmax=vmax)
		ax0.set_ylabel('h')
		ax0.set_yticks(np.arange(Ymse.shape[0]))
		ax0.set_yticklabels(sorted(Xmse))
		ax0.set_xlabel('lrG')
		ax0.set_xticks(np.arange(Ymse.shape[1]))
		ax0.set_xticklabels(sorted(Xmse[Xmse.keys()[0]]))
		ax0.set_title('Assimilation Error (MSE):\n Standard = {0} +/- {1}'.format(mse['standardAssimilation'][my_eps]['assim']['median'], mse['standardAssimilation'][my_eps]['assim']['std'] ))
		fig.colorbar(im, ax=ax0, extend='both')
		# im = ax1.imshow(Yt, interpolation='none')
		# ax1.set_xlabel('h')
		# ax1.set_xticks(np.arange(Yt.shape[0]))
		# ax1.set_xticklabels(sorted(Xt))
		# ax1.set_ylabel('lrG')
		# ax1.set_yticks(np.arange(Yt.shape[1]))
		# ax1.set_yticklabels(sorted(Xt[Xt.keys()[0]]))
		# ax1.set_title(r'Assimilation Time: Standard = {0} \pm {1}'.format(t_assim['standardAssimilation'][my_eps]['assim']['median'], t_assim['standardAssimilation'][my_eps]['assim']['std'] ))
		# fig.colorbar(im, ax=ax1, extend='both')

		fig.savefig(fname=output_dir+'/'+key_nm+'{0}_heatmap_method_comparison.png'.format(my_eps))
		plt.close(fig)

	### Plot G's across epsilon values
	fig, (axlist) = plt.subplots(nrows=1, ncols=len(G_all), figsize=[13,6], sharey=True)
	m = -1
	for method_nm in G_all:
		m += 1
		for kk in range(G.shape[1]):
			mean_vec = [G_all[method_nm][eps_val]['mean'][kk] for eps_val in eps_vec]
			std_vec = [G_all[method_nm][eps_val]['std'][kk] for eps_val in eps_vec]
			axlist[m].errorbar(x=eps_vec, y=mean_vec, yerr=std_vec, label='G_{0}'.format(kk))
			axlist[m].set_title(method_nm)
			axlist[m].set_xlabel(r'$\epsilon$ model error')

	axlist[0].legend()
	fig.suptitle('3DVAR Gain Learning')
	fig.savefig(fname=output_dir+'/'+key_nm+'_learned_gains.png')
	plt.close(fig)

	### Compare eps=0 case
	# method_vec = mse.keys()
	# for my_eps in eps_vec:
	# 	fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2)
	# 	pdb.set_trace()
	# 	median_vec = [mse[method_nm][my_eps]['assim']['median'] for method_nm in method_vec]
	# 	std_vec = [mse[method_nm][my_eps]['assim']['std'] for method_nm in method_vec]
	# 	ax0.bar(np.arange(len(method_vec)), median_vec, yerr=std_vec, align='center')
	# 	ax0.set_xticks(np.arange(len(method_vec)))
	# 	ax0.set_xticklabels(method_vec)
	# 	ax0.tick_params(axis='x', labelrotation=45)
	# 	ax0.set_title('Assimilation Error')
	# 	ax0.set_ylabel('MSE')

	# 	median_vec = [t_assim[method_nm][my_eps]['assim']['median'] for method_nm in method_vec]
	# 	std_vec = [t_assim[method_nm][my_eps]['assim']['std'] for method_nm in method_vec]
	# 	ax1.bar(np.arange(len(method_vec)), median_vec, yerr=std_vec, align='center')
	# 	ax1.set_xticks(np.arange(len(method_vec)))
	# 	ax1.set_xticklabels(method_vec)
	# 	ax1.xticks(rotation=45)
	# 	ax1.set_title('Assimilation Time')
	# 	ax1.set_ylabel('t_assim')

	# 	fig.suptitle(r'3DVAR Testing Performance for $\epsilon =$ {0}: Method Comparison'.format(my_eps))
	# 	fig.savefig(fname=output_dir+'/'+key_nm+'{0}_barChart_method_comparison.png'.format(my_eps))


if __name__ == '__main__':
	general_summary(my_dirs=MY_DIRS, output_dir=FLAGS.output_dir, n_train_trajectories=FLAGS.n_train_trajectories, n_test_trajectories=FLAGS.n_test_trajectories)
	epsilon_summary(my_dirs=MY_DIRS, output_dir=FLAGS.output_dir, n_train_trajectories=FLAGS.n_train_trajectories, n_test_trajectories=FLAGS.n_test_trajectories, key_nm='eps')

