## This script does the following
#1. Generates a random input sequence x
#2. Simulates data using a driven exponential decay ODE model
#3. Trains a single-layer RNN using clean data output from ODE and the input sequence
#4. RESULT: Gradients quickly go to 0 when training the RNN

# based off of code from https://www.cpuheater.com/deep-learning/introduction-to-recurrent-neural-networks-in-pytorch/
import os
import numpy as np
import numpy.matlib
from scipy.integrate import odeint
import torch
from torch.autograd import Variable
import torch.nn.init as init
import torch.cuda

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb

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

def run_ode_model(model, tspan, sim_model_params, tau=50, noise_frac=0, output_dir=".", drive_system=True):
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

	## Plot clean ODE simulation
	fig, (ax_list) = plt.subplots(y_clean.shape[1],1)
	if not isinstance(ax_list,np.ndarray):
		ax_list = [ax_list]
	for kk in range(len(ax_list)):
		ax = ax_list[kk]
		ax.plot(tspan, y_clean[:,kk], label='clean data')
		ax.scatter(tspan, y_noisy[:,kk], color='red', s=10, alpha=0.3, label='noisy data')
		ax.set_xlabel('time')
		ax.set_ylabel(sim_model_params['state_names'][kk] +'(t)')
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


def make_RNN_data(model, tspan, sim_model_params, noise_frac=0, output_dir=".", drive_system=True):

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	y_clean, y_noisy, x  = run_ode_model(model, tspan, sim_model_params, noise_frac=noise_frac, output_dir=output_dir, drive_system=drive_system)

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
def forward_vanilla(data_input, hidden_state, w1, w2, b, c, v, *args):
	hidden_state = torch.relu(b + torch.mm(w2,data_input) + torch.mm(w1,hidden_state))
	out = c + torch.mm(v,hidden_state)
	return  (out, hidden_state)

def forward_chaos_pureML(data_input, hidden_state, A, B, C, a, b, *args):
	hidden_state = torch.relu(a + torch.mm(A,hidden_state) + torch.mm(B,data_input))
	# hidden_state = torch.relu(a + torch.mm(A,hidden_state))
	out = b + torch.mm(C,hidden_state)
	return  (out, hidden_state)

def forward_chaos_pureML2(data_input, hidden_state, A, B, C, a, b, *args):
	hidden_state = torch.tanh(a + torch.mm(A,hidden_state))
	out = b + torch.mm(C,hidden_state)
	return  (out, hidden_state)


def forward_chaos_hybrid_full(model_input, hidden_state, A, B, C, a, b, normz_info, model, model_params):
	# unnormalize
	# ymin = normz_info['Ymin']
	# ymax = normz_info['Ymax']
	# ymean = normz_info['Ymean']
	# ysd = normz_info['Ysd']
	# xmean = normz_info['Xmean']
	# xsd = normz_info['Xsd']

	# y0 = ymean + hidden_state[0].detach().numpy()*ysd
	# y0 = ymin + ( hidden_state[0].detach().numpy()*(ymax - ymin) )
	tspan = [0, 0.5*model_params['delta_t'], model_params['delta_t']]
	# driver = xmean + xsd*model_input.detach().numpy()
	# my_args = model_params + (driver,)
	#
	# unnormalize model_input so that it can go through the ODE solver
	y0_normalized = torch.squeeze(model_input).detach()
	y0 = f_unNormalize_minmax(normz_info, y0_normalized.numpy())
	y_out = odeint(model, y0, tspan, args=model_params['ode_params'])

	y_pred = y_out[-1,:] #last column
	y_pred_normalized = f_normalize_minmax(normz_info, y_pred)

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

def train_chaosRNN(forward,
			y_clean_train, y_noisy_train,
			y_clean_test, y_noisy_test,
			model_params, hidden_size=6, n_epochs=100, lr=0.05,
			output_dir='.', normz_info=None, model=None,
			trivial_init=False,
			stack_hidden=True, stack_output=True,
			x_train=None, x_test=None,
			f_normalize_Y=f_normalize_minmax,
			f_unNormalize_Y=f_unNormalize_minmax,
			f_normalize_X = f_normalize_ztrans,
			f_unNormalize_X = f_unNormalize_ztrans,
			max_plot=2000):

	if torch.cuda.is_available():
		print('Using CUDA FloatTensor')
		dtype = torch.cuda.FloatTensor
	else:
		print('Using regular torch.FloatTensor')
		dtype = torch.FloatTensor

	n_plttrain = y_clean_train.shape[0] - min(max_plot,y_clean_train.shape[0])
	n_plttest = y_clean_test.shape[0] - min(max_plot,y_clean_test.shape[0])

	if hidden_size <= 50:
		keep_param_history = True
	else:
		keep_param_history = False

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	print('Starting RNN training for: ', output_dir)

	output_train = torch.FloatTensor(y_noisy_train).type(dtype)
	output_clean_train = torch.FloatTensor(y_clean_train).type(dtype)
	output_test = torch.FloatTensor(y_noisy_test).type(dtype)
	output_clean_test = torch.FloatTensor(y_clean_test).type(dtype)

	output_size = output_train.shape[1]
	train_seq_length = output_train.size(0)
	test_seq_length = output_test.size(0)

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
		pred = output_clean_test[0,:,None]
		predictions[0,:] = np.squeeze(pred)
		for i in range(test_seq_length-1):
			(pred, hidden_state) = forward(pred, hidden_state, A,B,C,a,b , normz_info, model, model_params)
			# hidden_state = hidden_state
			predictions[i+1,:] = pred.data.numpy().ravel()

		# plot predictions vs truth
		fig, (ax_list) = plt.subplots(y_clean_train.shape[1],1)
		if not isinstance(ax_list,np.ndarray):
			ax_list = [ax_list]

		for kk in range(len(ax_list)):
			ax1 = ax_list[kk]
			ax1.scatter(np.arange(len(y_noisy_test[n_plttest:,kk])), y_noisy_test[n_plttest:,kk], color='red', s=10, alpha=0.3, label='noisy data')
			ax1.plot(y_clean_test[n_plttest:,kk], color='red', label='clean data')
			ax1.plot(predictions[n_plttest:,kk], ':' ,color='red', label='NN trivial fit')
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
	else:
		init.normal_(A, 0.0, 0.1)
		init.normal_(B, 0.0, 0.1)
		init.normal_(C, 0.0, 0.1)
		init.normal_(a, 0.0, 0.1)
		init.normal_(b, 0.0, 0.1)

	A = Variable(A, requires_grad=True)
	B =  Variable(B, requires_grad=True)
	C =  Variable(C, requires_grad=True)
	a =  Variable(a, requires_grad=True)
	b =  Variable(b, requires_grad=True)

	if keep_param_history:
		A_history = np.zeros((n_epochs*(train_seq_length-1),A.shape[0],A.shape[1]))
		B_history = np.zeros((n_epochs*(train_seq_length-1),B.shape[0],B.shape[1]))
		C_history = np.zeros((n_epochs*(train_seq_length-1),C.shape[0],C.shape[1]))
		a_history = np.zeros((n_epochs*(train_seq_length-1),a.shape[0],a.shape[1]))
		b_history = np.zeros((n_epochs*(train_seq_length-1),b.shape[0],b.shape[1]))

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
		pred = output_train[0,:,None]
		hidden_state = Variable(torch.zeros((hidden_size, 1)).type(dtype), requires_grad=True)
		init.normal_(hidden_state,0.0,0.1)
		for j in range(train_seq_length-1):
			cc += 1
			target = output_train[j+1,None]
			target_clean = output_clean_train[j+1,None]
			(pred, hidden_state) = forward(output_train[j,:,None], hidden_state, A,B,C,a,b, normz_info, model, model_params)
			# (pred, hidden_state) = forward(pred.detach(), hidden_state, A,B,C,a,b, normz_info, model, model_params)
			loss = (pred.squeeze() - target.squeeze()).pow(2).sum()/2
			total_loss_train += loss
			total_loss_clean_train += (pred.squeeze() - target_clean.squeeze()).pow(2).sum()/2
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

			# save updated parameters
			if keep_param_history:
				A_history[cc,:] = A.detach().numpy()
				B_history[cc,:] = B.detach().numpy()
				C_history[cc,:] = C.detach().numpy()
				a_history[cc,:] = a.detach().numpy()
				b_history[cc,:] = b.detach().numpy()

		#normalize losses
		total_loss_train = total_loss_train / train_seq_length
		total_loss_clean_train = total_loss_clean_train / train_seq_length
		#store losses
		loss_vec_train[i_epoch] = total_loss_train
		loss_vec_clean_train[i_epoch] = total_loss_clean_train

		total_loss_test = 0
		total_loss_clean_test = 0
		# hidden_state = Variable(torch.zeros((hidden_size, 1)).type(dtype), requires_grad=False)
		pred = output_train[-1,:,None]
		for j in range(test_seq_length):
			target = output_test[j,None]
			target_clean = output_clean_test[j,None]
			(pred, hidden_state) = forward(pred, hidden_state, A,B,C,a,b, normz_info, model, model_params)
			total_loss_test += (pred.squeeze() - target.squeeze()).pow(2).sum()/2
			total_loss_clean_test += (pred.squeeze() - target_clean.squeeze()).pow(2).sum()/2
			pred = pred.detach()
			hidden_state = hidden_state.detach()
		#normalize losses
		total_loss_test = total_loss_test / test_seq_length
		total_loss_clean_test = total_loss_clean_test / test_seq_length
		#store losses
		loss_vec_test[i_epoch] = total_loss_test
		loss_vec_clean_test[i_epoch] = total_loss_clean_test

		# print updates every 10 iterations or in 10% incrememnts
		if i_epoch % int( max(2, np.ceil(n_epochs/10)) ) == 0:
			print("Epoch: {}\nTraining Loss = {}\nTesting Loss = {}".format(
						i_epoch,
						total_loss_train.data.item(),
						total_loss_test.data.item()))
				 # plot predictions vs truth
			# fig, (ax1, ax3) = plt.subplots(1, 2)
			fig, (ax_list) = plt.subplots(y_clean_train.shape[1],1)
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
				(pred, hidden_state) = forward(output_train[i,:,None], hidden_state, A,B,C,a,b, normz_info, model, model_params)
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
				t_plot = np.arange(0,round(len(y_noisy_train[n_plttrain:,kk])*model_params['delta_t'],8),model_params['delta_t'])
				ax1.scatter(t_plot, y_noisy_train_raw[n_plttrain:,kk], color='red', s=10, alpha=0.3, label='noisy data')
				ax1.plot(t_plot, y_clean_train_raw[n_plttrain:,kk], color='red', label='clean data')
				ax1.plot(t_plot, predictions_raw[n_plttrain:,kk], color='black', label='NN fit')
				ax1.set_xlabel('time')
				ax1.set_ylabel(model_params['state_names'][kk] + '(t)', color='red')
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
				t_plot = np.arange(0,round(len(saved_hidden_states[n_plttrain:,kk])*model_params['delta_t'],8),model_params['delta_t'])
				ax1.plot(t_plot, saved_hidden_states[n_plttrain:,kk], color='red', label='clean data')
				ax1.set_xlabel('time')
				ax1.set_ylabel('h_{}'.format(kk), color='red')
				ax1.tick_params(axis='y', labelcolor='red')

			ax_list[0].legend()

			fig.suptitle('Hidden State Dynamics')
			fig.savefig(fname=output_dir+'/rnn_train_hidden_states_iterEpochs'+str(i_epoch))
			plt.close(fig)


			fig, (ax_list) = plt.subplots(y_clean_train.shape[1],1)
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
				t_plot = np.arange(0,len(y_clean_test[n_plttest:,kk])*model_params['delta_t'],model_params['delta_t'])
				# ax3.scatter(np.arange(len(y_noisy_test)), y_noisy_test, color='red', s=10, alpha=0.3, label='noisy data')
				ax3.plot(t_plot, y_clean_test_raw[n_plttest:,kk], color='red', label='clean data')
				ax3.plot(t_plot, predictions_raw[n_plttest:,kk], color='black', label='NN fit')
				ax3.set_xlabel('time')
				ax3.set_ylabel(model_params['state_names'][kk] +'(t)', color='red')
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
				t_plot = np.arange(0,round(len(saved_hidden_states[n_plttest:,kk])*model_params['delta_t'],8),model_params['delta_t'])
				ax1.plot(t_plot, saved_hidden_states[n_plttest:,kk], color='red', label='clean data')
				ax1.set_xlabel('time')
				ax1.set_ylabel('h_{}'.format(kk), color='red')
				ax1.tick_params(axis='y', labelcolor='red')

			ax_list[0].legend()

			fig.suptitle('Hidden State Dynamics')
			fig.savefig(fname=output_dir+'/rnn_test_hidden_states_iterEpochs'+str(i_epoch))
			plt.close(fig)



			# Plot parameter convergence


	## save loss_vec
	np.savetxt(output_dir+'/loss_vec_train.txt',loss_vec_train)
	np.savetxt(output_dir+'/loss_vec_clean_train.txt',loss_vec_clean_train)
	np.savetxt(output_dir+'/loss_vec_test.txt',loss_vec_test)
	np.savetxt(output_dir+'/loss_vec_clean_test.txt',loss_vec_clean_test)
	np.savetxt(output_dir+'/A.txt',A.detach().numpy())
	np.savetxt(output_dir+'/B.txt',B.detach().numpy())
	np.savetxt(output_dir+'/C.txt',C.detach().numpy())
	np.savetxt(output_dir+'/a.txt',a.detach().numpy())
	np.savetxt(output_dir+'/b.txt',b.detach().numpy())
	# print("W1:",w1)
	# print("W2:",w2)
	# print("b:",b)
	# print("c:",c)
	# print("v:",v)

	# plot parameter convergence
	if keep_param_history:
		fig, my_axes = plt.subplots(2, 3, sharex=True, figsize=[8,6])

		x_vals = np.linspace(0,n_epochs,len(A_history))
		my_axes[0,0].plot(x_vals, np.linalg.norm(A.detach().numpy() - A_history, ord='fro', axis=(1,2)))
		my_axes[0,0].set_title('A')
		my_axes[0,0].set_xlabel('Epochs')

		my_axes[0,1].plot(x_vals, np.linalg.norm(B.detach().numpy() - B_history, ord='fro', axis=(1,2)))
		my_axes[0,1].set_title('B')
		my_axes[0,1].set_xlabel('Epochs')

		my_axes[1,0].plot(x_vals, np.linalg.norm(C.detach().numpy() - C_history, ord='fro', axis=(1,2)))
		my_axes[1,0].set_title('C')
		my_axes[1,0].set_xlabel('Epochs')

		my_axes[1,1].plot(x_vals, np.linalg.norm(a.detach().numpy() - a_history, ord='fro', axis=(1,2)))
		my_axes[1,1].set_title('a')
		my_axes[1,1].set_xlabel('Epochs')

		my_axes[1,2].plot(x_vals, np.linalg.norm(b.detach().numpy() - b_history, ord='fro', axis=(1,2)))
		my_axes[1,2].set_title('b')
		my_axes[1,2].set_xlabel('Epochs')


		fig.suptitle("Parameter convergence")
		fig.savefig(fname=output_dir+'/rnn_parameter_convergence')
		plt.close(fig)

	## now, inspect the quality of the learned model
	# plot predictions vs truth
	# fig, (ax1, ax3) = plt.subplots(1, 2)
	fig, (ax_list) = plt.subplots(y_clean_train.shape[1],1)
	if not isinstance(ax_list,np.ndarray):
		ax_list = [ax_list]

	# first run and plot training fits
	hidden_state = torch.zeros((hidden_size, 1)).type(dtype)
	init.normal_(hidden_state,0.0,0.1)
	predictions = np.zeros([train_seq_length, output_size])
	pred = output_train[0,:,None]
	predictions[0,:] = np.squeeze(output_train[0,:,None])
	for i in range(train_seq_length-1):
		(pred, hidden_state) = forward(output_train[i,:,None], hidden_state, A,B,C,a,b, normz_info, model, model_params)
		# (pred, hidden_state) = forward(pred, hidden_state, A,B,C,a,b, normz_info, model, model_params)
		# hidden_state = hidden_state
		predictions[i+1,:] = pred.data.numpy().ravel()

	predictions_raw = f_unNormalize_Y(normz_info,predictions)
	for kk in range(len(ax_list)):
		ax1 = ax_list[kk]
		t_plot = np.arange(0,round(len(y_noisy_train_raw[n_plttrain:,kk])*model_params['delta_t'],8),model_params['delta_t'])
		ax1.scatter(t_plot, y_noisy_train_raw[n_plttrain:,kk], color='red', s=10, alpha=0.3, label='noisy data')
		ax1.plot(t_plot, y_clean_train_raw[n_plttrain:,kk], color='red', label='clean data')
		ax1.plot(t_plot, predictions_raw[n_plttrain:,kk], color='black', label='NN fit')
		ax1.set_xlabel('time')
		ax1.set_ylabel(model_params['state_names'][kk] +'(t)', color='red')
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

	fig, (ax_list) = plt.subplots(y_clean_train.shape[1],1)
	if not isinstance(ax_list,np.ndarray):
		ax_list = [ax_list]
	predictions_raw = f_unNormalize_Y(normz_info,predictions)
	for kk in range(len(ax_list)):
		ax3 = ax_list[kk]
		t_plot = np.arange(0,len(y_clean_test_raw[n_plttest:,kk])*model_params['delta_t'],model_params['delta_t'])
		# ax3.scatter(np.arange(len(y_noisy_test)), y_noisy_test, color='red', s=10, alpha=0.3, label='noisy data')
		ax3.plot(t_plot, y_clean_test_raw[n_plttest:,kk], color='red', label='clean data')
		ax3.plot(t_plot, predictions_raw[n_plttest:,kk], color='black', label='NN fit')
		ax3.set_xlabel('time')
		ax3.set_ylabel(model_params['state_names'][kk] +'(t)', color='red')
		ax3.tick_params(axis='y', labelcolor='red')
		# ax3.set_title('Testing Fit')

	ax_list[0].legend()
	fig.suptitle('RNN TEST fit to ODE simulation')
	fig.savefig(fname=output_dir+'/rnn_fit_ode_TEST')
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


def compare_fits(my_dirs, output_fname="./training_comparisons"):
	fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
		figsize = [10, 4],
		sharey=True, sharex=True)
	for d in my_dirs:
			d_label = d.split("/")[-1].rstrip('_noisy').rstrip('_clean')
			x = np.loadtxt(d+"/loss_vec_train.txt")
			ax1.plot(x, label=d_label)
			x = np.loadtxt(d+"/loss_vec_clean_test.txt")
			ax2.plot(x, label=d_label)
			# x = np.loadtxt(d+"/loss_vec_test.txt")
			# ax3.plot(x, label=d_label)
	ax1.set_xlabel('Epochs')
	ax1.set_ylabel('Error')
	ax1.set_title('Train Error')
	ax1.legend(fontsize=6, handlelength=2, loc='upper right')
	ax2.set_xlabel('Epochs')
	ax2.set_ylabel('Error')
	ax2.set_title('Test Error (predicting clean data)')
	# ax3.set_xlabel('Epochs')
	# ax3.set_ylabel('Error')
	# ax3.set_title('Test Error (on noisy data)')
	# fig.suptitle("Comparison of training efficacy (trained on noisy data)")
	fig.savefig(fname=output_fname)
	# plot in log scale
	ax1.set_yscale('log')
	ax2.set_yscale('log')
	ax1.set_ylabel('log Error')
	ax2.set_ylabel('log Error')
	fig.savefig(fname=output_fname+'_log')
	plt.close(fig)




