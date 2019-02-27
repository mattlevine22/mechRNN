## This script does the following
#1. Generates a random input sequence x
#2. Simulates data using a driven exponential decay ODE model
#3. Trains a single-layer RNN using clean data output from ODE and the input sequence
#4. RESULT: Gradients quickly go to 0 when training the RNN

# based off of code from https://www.cpuheater.com/deep-learning/introduction-to-recurrent-neural-networks-in-pytorch/
import torch
from torch.autograd import Variable
import numpy as np
#import pylab as pl
import torch.nn.init as init
from scipy.integrate import odeint
import pdb
import matplotlib.pyplot as plt

# fix random seeds for deterministic behavior of this script
np.random.seed(7)
torch.manual_seed(7)

### ODE simulation section
## 1. Simulate ODE
# function that returns dy/dt
def model(y,t,yb,c_gamma,x):
  # pdb.set_trace()
  x_t = x[np.where(x[:,0] <= t)[0][-1], 1]
  dydt = -c_gamma*(y-yb) + x_t
  return dydt

# time points
tspan = np.arange(0,1001,1)
yb = 100
c_gamma = 0.05
tau = 50 # window length of persistence
tmp = np.arange(0,1000,tau)
x = np.zeros([len(tmp),2])
x[:,0] = tmp
x[:,1] = 10*np.random.rand(len(x))
y0 = yb
y = odeint(model, y0, tspan, args=(yb,c_gamma,x))/100.0

## 2. Plot ODE
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.plot(tspan, y, color=color)
ax1.set_xlabel('time')
ax1.set_ylabel('y(t)', color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('x(t)', color=color)  # we already handled the x-label with ax1
ax2.step(x[:,0], x[:,1], ':', where='post', color=color, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color)

fig.suptitle('Driven Exponential Decay simulation with simple inputs')
fig.savefig(fname='driven_exponential_decay_simpleInputs')

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
input = torch.FloatTensor(x0)
output = torch.FloatTensor(y)

### RNN fitting section
def forward(input, hidden_state, w1, w2, b, c, v):
  # pdb.set_trace()
  hidden_state = torch.tanh(b + torch.mm(w2,input) + torch.mm(w1,hidden_state))
  out = c + torch.mm(v,hidden_state)
  return  (out, hidden_state)

dtype = torch.FloatTensor
input_size, hidden_size, output_size = 1, 6, 1
epochs = 40
seq_length = x0.shape[0]
lr = 0.02

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

for i in range(epochs):
  total_loss = 0
  #init.normal_(hidden_state, 0.0, 1)
  #hidden_state = Variable(hidden_state, requires_grad=True)
  hidden_state = Variable(torch.zeros((hidden_size, 1)).type(dtype), requires_grad=True)
  for j in range(seq_length):
    target = output[j:(j+1)]
    (pred, hidden_state) = forward(input[j:j+1], hidden_state, w1, w2, b, c, v)
    loss = (pred - target).pow(2).sum()/2
    total_loss += loss
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
  if i % 10 == 0:
     print("Epoch: {}. Output-loss {}".format(
          i,
          total_loss.data.item()))


## now, inspect the quality of the learned model
hidden_state = Variable(torch.zeros((hidden_size, 1)).type(dtype), requires_grad=False)
predictions = []
for i in range(seq_length):
  (pred, hidden_state) = forward(input[i:i+1], hidden_state, w1, w2, b, c, v)
  hidden_state = hidden_state
  predictions.append(pred.data.numpy().ravel()[0])

# plot predictions vs truth
fig, ax1 = plt.subplots()

ax1.plot(output.numpy(), color='red')
ax1.plot(predictions, ':' ,color='red')
ax1.set_xlabel('time')
ax1.set_ylabel('y(t)', color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('x(t)', color=color)  # we already handled the x-label with ax1
ax2.plot((input.numpy()), ':', color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.suptitle('RNN fit to Exponential Decay simulation')
fig.savefig(fname='rnn_fit_ode')

