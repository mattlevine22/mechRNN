import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import pdb




class RNN_VANILLA(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, cell_type, embed_physics_prediction, use_physics_as_bias, dtype, tsynch, teacher_force_probability):
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

    def normalize(x):
        return x

    def unnormalize(x):
        return x

    def get_physics_prediction(x0):
        # self.delta_t
        return x0

    def forward(self, input_states, physical_predictions=None, train=False):
        rnn_preds = [] #output of the RNN (i.e. residual)
        full_preds = [] #final output prediction (i.e. Psi0(x_t) + RNN(x_t,h_t))

        #useful link for teacher-forcing: https://towardsdatascience.com/lstm-for-time-series-prediction-de8aeb26f2ca
        #

        # input_states corresponds to data x_t
        # physical_predictions corresponds to predictions \Psi_0(x_t), i.e. \hat{x_t}
        # optional bias term driven by glm_scores

        if self.embed_physics_prediction:
            input_sequence = torch.stack(input_states, physical_predictions)
        else:
            input_sequence = input_states


        # initialize states
        h_t = torch.zeros(input_sequence.size(0), self.hidden_size, dtype=torch.FloatTensor.type(self.dtype))
        c_t = torch.zeros(input_sequence.size(0), self.hidden_size, dtype=torch.FloatTensor.type(self.dtype))

        # consider Scheduled Sampling (https://arxiv.org/abs/1506.03099) where probability of using RNN-output increases as you train.
        for t in range(input_sequence.shape[0]):
            # get input to hidden state
            if train:
                # consider teacher forcing (using RNN output prediction as next training input instead of training data)
                if t>self.t_synch and random.random()<self.teacher_force_probability:
                    input_t = full_pred #feed RNN prediction back in as next input
                else:
                    input_t = input_sequence[t]
            else:
                physics_pred = self.get_physics_prediction(x0=self.unnormalize(full_pred), dt=self.delta_t)
                input_t = torch.stack(pred_output, self.normalize(physics_prediction))

            # evolve hidden state
            h_t, c_t = self.cell(input_t, (h_t, c_t))
            rnn_pred = self.hidden2pred(h_t)
            full_pred = self.use_physics_as_bias * physical_predictions[t,:] + rnn_pred
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
    if name is 'nn.MSELoss':
        return nn.MSELoss()
    else:
        raise('Loss name not recognized')

def get_model(input_size, name='RNN_VANILLA', hidden_size=50, output_size=None, cell_type='LSTM', embed_physics_prediction=False, dtype=torch.FloatTensor, t_synch=1000, teacher_force_probability=0.0):
    if output_size is None:
        output_size = input_size

    if name=='RNN_VANILLA':
        return RNN_VANILLA(input_size=input_size, hidden_size=hidden_size, output_size=output_size, cell_type=cell_type, dtype=dtype)
    else:
        return None
