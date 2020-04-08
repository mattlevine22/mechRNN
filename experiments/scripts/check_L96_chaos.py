# Correspondence with Dima via Whatsapp on Feb 24, 2020:
# RK45 (explicit) for slow-system-only
# RK45 (implicit) aka Radau for multi-scale-system
# In both cases, set abstol to 1e-6, reltol to 1e-3, dtmax to 1e-3

from odelibrary import L96M #(from file import class)
import os
import json
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb

def getval(x):
	try:
		return getattr(x, '__dict__', str(x))
	except:
		return None

def run_ode_model(model, tspan, sim_model_params, output_dir=".", plot_slow_indices=None, plot_fast_indices=None):

	my_args = sim_model_params['ode_params']
	y0 = sim_model_params['state_init']
	sol = solve_ivp(fun=lambda t, y: model(y, t, *my_args), t_span=(tspan[0], tspan[-1]), y0=np.array(y0).T, method=sim_model_params['ode_int_method'], rtol=sim_model_params['ode_int_rtol'], atol=sim_model_params['ode_int_atol'], max_step=sim_model_params['ode_int_max_step'], t_eval=tspan)
	y_clean = sol.y.T

	ind_dict = {'slow': plot_slow_indices, 'fast': plot_fast_indices}
	for ind_key in ind_dict:
		plot_inds = ind_dict[ind_key]
		if plot_inds is None:
			continue
		## Plot clean ODE simulation
		fig, ax_list = plt.subplots(len(plot_inds),len(plot_inds), figsize=[11,11])
		for i_y in range(len(plot_inds)):
			yy = plot_inds[i_y]
			left_y = True
			if yy == plot_inds[-1]:
				bottom_x = True
			else:
				bottom_x = False
			for i_x in range(len(plot_inds)):
				xx = plot_inds[i_x]

				ax = ax_list[i_y][i_x]
				if xx!=yy:
					ax.plot(y_clean[:,xx],y_clean[:,yy])
				if bottom_x:
					ax.set_xlabel(sim_model_params['state_names'][xx])
				if left_y:
					ax.set_ylabel(sim_model_params['state_names'][yy])
					left_y = False
		fig.suptitle('ODE simulation of '+ind_key+' variables')
		fig.savefig(fname=output_dir+'/ODEsimulation_' + ind_key)
		plt.close(fig)

	return y_clean


def make_plots(sim_model_params=dict(), output_dir='.', K=4, J=4, F=10, eps_power=-7, delta_t=0.01, T=10, decoupled=False):
	# delta_t = 0.01 # output interval for ODE simulation
	# T = 10 # Total length of ODE simulation
	# K = 4
	# J = 4
	eps = 2**(eps_power)
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	# write parameters to file
	settings_fname = output_dir + '/run_settings.txt'
	with open(settings_fname, 'w') as f:
		json.dump(locals(), f, default=lambda x: getval(x) , indent=2)

	tspan = np.arange(0,T,delta_t)
	plot_fast_indices = [K, K+1, K+J-2, K+J-1]
	plot_slow_indices = np.arange(K)

	l96m = L96M(K=K, J=J, F=F, eps=eps)
	# establish initial conditions
	state_init = np.squeeze(l96m.get_inits())

	if decoupled:
		state_init = state_init[K:] # only fast variables
		sim_model = l96m.decoupled
		plot_slow_indices = None
	else:
		sim_model = l96m.full

	# set up state names
	state_names = ['X_'+ str(k+1) for k in range(K)]
	for k in range(K):
		state_names += ['Y_' + str(j+1) + ',' + str(k+1) for j in range(J)]

	sim_model_params['state_names'] = state_names
	sim_model_params['state_init'] = state_init

	run_ode_model(sim_model, tspan, sim_model_params, output_dir=output_dir, plot_slow_indices=plot_slow_indices, plot_fast_indices=plot_fast_indices)
	return

def make_traj_plots(n_inits=2, sd_perturb=0.01, sim_model_params=dict(), output_dir='.', K=4, J=4, F=50, eps_power=-7, delta_t=0.01, T=5, decoupled=False):
	# delta_t = 0.01 # output interval for ODE simulation
	# T = 10 # Total length of ODE simulation
	# K = 4
	# J = 4

	# set up state names
	state_names = ['X_'+ str(k+1) for k in range(K)]
	for k in range(K):
		state_names += ['Y_' + str(j+1) + ',' + str(k+1) for j in range(J)]
	sim_model_params['state_names'] = state_names

	eps = 2**(eps_power)
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

	# write parameters to file
	settings_fname = output_dir + '/run_settings.txt'
	with open(settings_fname, 'w') as f:
		json.dump(locals(), f, default=lambda x: getval(x) , indent=2)

	tspan = np.arange(0,T,delta_t)
	plot_fast_indices = [K, K+1, K+J-2, K+J-1]
	plot_slow_indices = np.arange(K)

	l96m = L96M(K=K, J=J, F=F, eps=eps)
	sim_model = l96m.full
	# establish initial conditions
	state_init = np.squeeze(l96m.get_inits())


	Y = []
	sim_model_params['state_init'] = state_init
	y_clean = run_ode_model(sim_model, tspan, sim_model_params, output_dir=output_dir, plot_slow_indices=plot_slow_indices, plot_fast_indices=plot_fast_indices)
	Y.append(y_clean)

	sim_model_params['state_init'] = state_init + sd_perturb*np.random.randn()
	y_clean_perturbed = run_ode_model(sim_model, tspan, sim_model_params, output_dir=output_dir, plot_slow_indices=plot_slow_indices, plot_fast_indices=plot_fast_indices)
	Y.append(y_clean_perturbed)

	fig, ax_list = plt.subplots(len(plot_slow_indices), 1, figsize=[11,11])
	for k in range(K):
		ax = ax_list[k]
		for y in Y:
			ax.plot(delta_t*np.arange(y.shape[0]), y[:,k])
		ax.set_ylabel(sim_model_params['state_names'][k])
		ax.set_xlabel('time')
	fig.savefig(fname=os.path.join(output_dir,'ODEsensitivity'))
	plt.close(fig)

	return

if __name__ == '__main__':

	sim_model_params = {'ode_params': (),
						'ode_int_method':'Radau',
						'ode_int_rtol':1e-3,
						'ode_int_atol':1e-6,
						'ode_int_max_step':1e-3}

	main_dir = '/Users/matthewlevine/Downloads/l96_check_chaos_NEW'
	if not os.path.exists(main_dir):
		os.mkdir(main_dir)

	T = 10
	eps_power = -7
	F = 50
	output_dir = main_dir+'/l96chaos_full_F{0}_epsPower{1}_T{2}'.format(F,eps_power,T)
	make_traj_plots(n_inits=2, sd_perturb=0.5, sim_model_params=sim_model_params, output_dir=output_dir, F=F, eps_power=eps_power, T=T, decoupled=False)


	# T_list = [20,50]
	# F_list = [18,22,10,15,25,50,1,5,100]
	# eps_power_list = [-1, -3, -5, -7, -9]

	# for T in T_list:
	# 	print('T=',T)
	# 	for F in F_list: #[10,25,50]:
	# 		print('F=',F)
	# 		if T==T_list[0]:
	# 			output_dir = main_dir+'/l96chaos_decoupled_F{0}'.format(F)
	# 			if not os.path.exists(output_dir):
	# 				make_plots(sim_model_params=sim_model_params, output_dir=output_dir, F=F, delta_t=0.0001, T=0.2, decoupled=True)
	# 		for eps_power in eps_power_list:
	# 			print('eps_power=',eps_power)
	# 			output_dir = main_dir+'/l96chaos_full_F{0}_epsPower{1}_T{2}'.format(F,eps_power,T)
	# 			if not os.path.exists(output_dir):
	# 				make_plots(sim_model_params=sim_model_params, output_dir=output_dir, F=F, eps_power=eps_power, T=T, decoupled=False)


