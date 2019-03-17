## This script does the following
#1. Generates a random input sequence x
#2. Simulates data using a driven exponential decay ODE model
#3. Trains a single-layer RNN using clean data output from ODE and the input sequence
#4. RESULT: Gradients quickly go to 0 when training the RNN

# based off of code from https://www.cpuheater.com/deep-learning/introduction-to-recurrent-neural-networks-in-pytorch/
import os
import numpy as np
from scipy.integrate import odeint
import torch
from torch.autograd import Variable
import torch.nn.init as init

import matplotlib.pyplot as plt

import pdb

### ODE simulation section
## 1. Simulate ODE
# function that returns dy/dt
def exp_decay_model(y,t,yb,c_gamma,x):
	x_t = x[np.where(x[:,0] <= t)[0][-1], 1]
	dydt = -c_gamma*(y-yb) + x_t
	return dydt

def easy_exp_decay_model(y_in,t,yb,c_gamma,x_in):
	dydt = -c_gamma*(y_in-yb) + x_in[0]
	return dydt


def run_ode_model(model, tspan, sim_model_params, tau=50, noise_frac=0, output_dir="."):
	# time points
	# tau = 50 # window length of persistence
	tmp = np.arange(0,1000,tau)
	x = np.zeros([len(tmp),2])
	x[:,0] = tmp
	x[:,1] = 10*np.random.rand(len(x))

	y0 = sim_model_params[0]
	my_args = sim_model_params + (x,)
	y_clean = odeint(model, y0, tspan, args=my_args)

	y_noisy = y_clean + noise_frac*np.mean(y_clean)*np.random.randn(len(y_clean),1)

	## 2. Plot ODE
	fig, ax1 = plt.subplots()

	color = 'tab:red'
	ax1.scatter(tspan, y_noisy, s=10, alpha=0.3, color=color, label='noisy simulation')
	ax1.plot(tspan, y_clean, color=color, label='clean simulation')
	ax1.set_xlabel('time')
	ax1.set_ylabel('y(t)', color=color)
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:blue'
	ax2.set_ylabel('x(t)', color=color)  # we already handled the x-label with ax1
	ax2.step(x[:,0], x[:,1], ':', where='post', color=color, linestyle='--', label='driver/input data')
	ax2.tick_params(axis='y', labelcolor=color)

	fig.legend()
	fig.suptitle('Driven Exponential Decay simulation with simple inputs')
	fig.savefig(fname=output_dir+'/driven_exponential_decay_simpleInputs')
	plt.close(fig)

	return y_clean, y_noisy, x


def make_RNN_data(model, tspan, sim_model_params, noise_frac=0, output_dir="."):

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	y_clean, y_noisy, x  = run_ode_model(model, tspan, sim_model_params, noise_frac=noise_frac, output_dir=output_dir)

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

	return input_data, y_clean, y_noisy


### RNN fitting section
def forward_vanilla(input, hidden_state, w1, w2, b, c, v, *args):
	hidden_state = torch.relu(b + torch.mm(w2,input) + torch.mm(w1,hidden_state))
	out = c + torch.mm(v,hidden_state)
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


def train_RNN(forward,
			y_clean_train, y_noisy_train, x_train,
			y_clean_test, y_noisy_test, x_test,
			model_params, hidden_size=6, n_epochs=100, lr=0.05,
			output_dir='.', normz_info=None, model=None):

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	print('Starting RNN training for: ', output_dir)

	x_train = torch.FloatTensor(x_train)
	x_test = torch.FloatTensor(x_test)

	output_train = torch.FloatTensor(y_noisy_train)
	output_clean_train = torch.FloatTensor(y_clean_train)
	output_test = torch.FloatTensor(y_noisy_test)
	output_clean_test = torch.FloatTensor(y_clean_test)

	dtype = torch.FloatTensor
	input_size, output_size = x_train.shape[0], output_train.shape[1]
	train_seq_length = output_train.size(0)
	test_seq_length = output_test.size(0)

	# first, SHOW that a simple mechRNN can fit the data perfectly
	# now, TRAIN to fit the output from the previous model
	w1 = torch.zeros(hidden_size, hidden_size).type(dtype)
	w1[0,0] = 1.
	w2 = torch.zeros(hidden_size, input_size).type(dtype)
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

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:blue'
	ax2.set_ylabel('x(t)', color=color)  # we already handled the x-label with ax1
	ax2.plot((x_test[0,:].numpy()), ':', color=color, linestyle='--', label='input/driver data')
	ax2.tick_params(axis='y', labelcolor=color)

	fig.legend()
	fig.suptitle('RNN w/ just mechanism fit to Exponential Decay simulation TEST SET')
	fig.savefig(fname=output_dir+'/PERFECT_MechRnn_fit_ode')
	plt.close(fig)

	# now, TRAIN to fit the output from the previous model
	w1 = torch.FloatTensor(hidden_size, hidden_size).type(dtype)
	init.normal_(w1, 0.0, 0.1)
	w1 =  Variable(w1, requires_grad=True)

	w2 = torch.FloatTensor(hidden_size, input_size).type(dtype)
	init.normal_(w2, 0.0, 0.1)
	w2 = Variable(w2, requires_grad=True)

	b = torch.FloatTensor(hidden_size, 1).type(dtype)
	init.normal_(b, 0.0, 0.1)
	b =  Variable(b, requires_grad=True)

	c = torch.FloatTensor(output_size, 1).type(dtype)
	init.normal_(c, 0.0, 0.1)
	c =  Variable(c, requires_grad=True)

	v = torch.FloatTensor(output_size, hidden_size).type(dtype)
	init.normal_(v, 0.0, 0.1)
	v =  Variable(v, requires_grad=True)

	loss_vec_train = np.zeros(n_epochs)
	loss_vec_clean_train = np.zeros(n_epochs)
	loss_vec_test = np.zeros(n_epochs)
	loss_vec_clean_test = np.zeros(n_epochs)
	for i_epoch in range(n_epochs):
		total_loss_train = 0
		total_loss_clean_train = 0
		#init.normal_(hidden_state, 0.0, 1)
		#hidden_state = Variable(hidden_state, requires_grad=True)
		hidden_state = Variable(torch.zeros((hidden_size, 1)).type(dtype), requires_grad=False)
		for j in range(train_seq_length):
			target = output_train[j:(j+1)]
			target_clean = output_clean_train[j:(j+1)]
			(pred, hidden_state) = forward(x_train[:,j:j+1], hidden_state, w1, w2, b, c, v, normz_info, model, model_params)
			loss = (pred - target).pow(2).sum()/2
			total_loss_train += loss
			total_loss_clean_train += (pred - target_clean).pow(2).sum()/2
			loss.backward()

			w1.data -= lr * w1.grad.data
			w2.data -= lr * w2.grad.data
			b.data -= lr * b.grad.data
			c.data -= lr * c.grad.data
			v.data -= lr * v.grad.data

			w1.grad.data.zero_()
			w2.grad.data.zero_()
			b.grad.data.zero_()
			c.grad.data.zero_()
			v.grad.data.zero_()

			hidden_state = hidden_state.detach()
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
			(pred, hidden_state) = forward(x_test[:,j:j+1], hidden_state, w1, w2, b, c, v, normz_info, model, model_params)
			total_loss_test += (pred - target).pow(2).sum()/2
			total_loss_clean_test += (pred - target_clean).pow(2).sum()/2

			hidden_state = hidden_state.detach()
		#normalize losses
		total_loss_test = total_loss_test / test_seq_length
		total_loss_clean_test = total_loss_clean_test / test_seq_length
		#store losses
		loss_vec_test[i_epoch] = total_loss_test
		loss_vec_clean_test[i_epoch] = total_loss_clean_test

		# print updates every iteration or in 10% incrememnts
		if i_epoch % int( max(1, np.ceil(n_epochs/10)) ) == 0:
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
				(pred, hidden_state) = forward(x_train[:,i:i+1], hidden_state, w1, w2, b, c, v, normz_info, model, model_params)
				hidden_state = hidden_state
				predictions.append(pred.data.numpy().ravel()[0])

			ax1.scatter(np.arange(len(y_noisy_train)), y_noisy_train, color='red', s=10, alpha=0.3, label='noisy data')
			ax1.plot(y_clean_train, color='red', label='clean data')
			ax1.plot(predictions, color='black', label='NN fit')
			ax1.set_xlabel('time')
			ax1.set_ylabel('y(t)', color='red')
			ax1.tick_params(axis='y', labelcolor='red')

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
				(pred, hidden_state) = forward(x_test[:,i:i+1], hidden_state, w1, w2, b, c, v, normz_info, model, model_params)
				hidden_state = hidden_state
				predictions.append(pred.data.numpy().ravel()[0])

			ax3.scatter(np.arange(len(y_noisy_test)), y_noisy_test, color='red', s=10, alpha=0.3, label='noisy data')
			ax3.plot(y_clean_test, color='red', label='clean data')
			ax3.plot(predictions, color='black', label='NN fit')
			ax3.set_xlabel('time')
			ax3.set_ylabel('y(t)', color='red')
			ax3.tick_params(axis='y', labelcolor='red')

			ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis

			color = 'tab:blue'
			ax4.set_ylabel('x(t)', color=color)  # we already handled the x-label with ax1
			ax4.plot((x_test[0,:].numpy()), ':', color=color, linestyle='--', label='driver/input data')
			ax4.tick_params(axis='y', labelcolor=color)

			ax3.set_title('Testing Fit')

			ax3.legend()
			ax2.legend()

			fig.suptitle('RNN fit to Exponential Decay simulation--' + str(i_epoch) + 'training epochs')
			fig.savefig(fname=output_dir+'/rnn_fit_ode_iterEpochs'+str(i_epoch))
			plt.close(fig)

	## save loss_vec
	np.savetxt(output_dir+'/loss_vec_train.txt',loss_vec_train)
	np.savetxt(output_dir+'/loss_vec_clean_train.txt',loss_vec_clean_train)
	np.savetxt(output_dir+'/loss_vec_test.txt',loss_vec_test)
	np.savetxt(output_dir+'/loss_vec_clean_test.txt',loss_vec_clean_test)
	np.savetxt(output_dir+'/w1.txt',w1.detach().numpy())
	np.savetxt(output_dir+'/w2.txt',w2.detach().numpy())
	np.savetxt(output_dir+'/b.txt',b.detach().numpy())
	np.savetxt(output_dir+'/c.txt',c.detach().numpy())
	np.savetxt(output_dir+'/v.txt',v.detach().numpy())

	# print("W1:",w1)
	# print("W2:",w2)
	# print("b:",b)
	# print("c:",c)
	# print("v:",v)

	## now, inspect the quality of the learned model

	# plot predictions vs truth
	fig, (ax1, ax3) = plt.subplots(1, 2)

	# first run and plot training fits
	hidden_state = Variable(torch.zeros((hidden_size, 1)).type(dtype), requires_grad=False)
	predictions = []
	for i in range(train_seq_length):
		(pred, hidden_state) = forward(x_train[:,i:i+1], hidden_state, w1, w2, b, c, v, normz_info, model, model_params)
		hidden_state = hidden_state
		predictions.append(pred.data.numpy().ravel()[0])

	ax1.scatter(np.arange(len(y_noisy_train)), y_noisy_train, color='red', s=10, alpha=0.3, label='noisy data')
	ax1.plot(y_clean_train, color='red', label='clean data')
	ax1.plot(predictions, color='black', label='NN fit')
	ax1.set_xlabel('time')
	ax1.set_ylabel('y(t)', color='red')
	ax1.tick_params(axis='y', labelcolor='red')

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
		(pred, hidden_state) = forward(x_test[:,i:i+1], hidden_state, w1, w2, b, c, v, normz_info, model, model_params)
		hidden_state = hidden_state
		predictions.append(pred.data.numpy().ravel()[0])

	ax3.scatter(np.arange(len(y_noisy_test)), y_noisy_test, color='red', s=10, alpha=0.3, label='noisy data')
	ax3.plot(y_clean_test, color='red', label='clean data')
	ax3.plot(predictions, color='black', label='NN fit')
	ax3.set_xlabel('time')
	ax3.set_ylabel('y(t)', color='red')
	ax3.tick_params(axis='y', labelcolor='red')

	ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:blue'
	ax4.set_ylabel('x(t)', color=color)  # we already handled the x-label with ax1
	ax4.plot((x_test[0,:].numpy()), ':', color=color, linestyle='--', label='driver/input data')
	ax4.tick_params(axis='y', labelcolor=color)

	ax3.set_title('Testing Fit')

	ax3.legend()
	ax2.legend()

	fig.suptitle('RNN fit to Exponential Decay simulation')
	fig.savefig(fname=output_dir+'/rnn_fit_ode')
	plt.close(fig)


def compare_fits(my_dirs, output_fname="./training_comparisons"):

	fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,
		figsize = [10, 4],
		sharey=True, sharex=True)
	for d in my_dirs:
			d_label = d.split("/")[-1].rstrip('_noisy').rstrip('_clean')
			x = np.loadtxt(d+"/loss_vec_train.txt")
			ax1.plot(x, label=d_label)
			x = np.loadtxt(d+"/loss_vec_clean_test.txt")
			ax2.plot(x, label=d_label)
			x = np.loadtxt(d+"/loss_vec_test.txt")
			ax3.plot(x, label=d_label)

	ax1.set_xlabel('Epochs')
	ax1.set_ylabel('Error')
	ax1.set_title('Train Error')
	ax1.legend(fontsize=6, handlelength=2, loc='upper right')
	ax2.set_xlabel('Epochs')
	ax2.set_ylabel('Error')
	ax2.set_title('Test Error (on clean data)')
	ax3.set_xlabel('Epochs')
	ax3.set_ylabel('Error')
	ax3.set_title('Test Error (on noisy data)')

	# fig.suptitle("Comparison of training efficacy (trained on noisy data)")
	fig.savefig(fname=output_fname)
	plt.close(fig)




