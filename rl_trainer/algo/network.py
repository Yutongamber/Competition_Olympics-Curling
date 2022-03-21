import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from .torch_utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


STD = 2**0.5

def initialize_weights(mod, initialization_type, scale=STD):
    '''
    Weight initializer for the models.
    Inputs: A model, Returns: none, initializes the parameters
    '''
    for p in mod.parameters():
        if initialization_type == "normal":
            p.data.normal_(0.01)
        elif initialization_type == "xavier":
            if len(p.data.shape) >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                p.data.zero_()
        elif initialization_type == "orthogonal":
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
            else:
                p.data.zero_()
        else:
            raise ValueError("Need a valid initialization key")

class Actor(nn.Module):
    def __init__(self, state_space, action_space,init_type, hidden_size=64):
        super(Actor, self).__init__()
        self.linear_in = nn.Linear(state_space, hidden_size)
        initialize_weights(self.linear_in,init_type)
        self.action_head = nn.Linear(hidden_size, action_space)
        initialize_weights(self.action_head,init_type)

    def forward(self, x):
        x = F.relu(self.linear_in(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob
    
    def entropies(self, p):
        '''
        p is probs of shape (batch_size, action_space). return mean entropy
        across the batch of states
        '''
        entropies = (p * ch.log(p)).sum(dim=1)
        return entropies

class Critic(nn.Module):
    def __init__(self, state_space,init_type, hidden_size=64):
        super(Critic, self).__init__()
        self.linear_in = nn.Linear(state_space, hidden_size)
        initialize_weights(self.linear_in,init_type)
        self.state_value = nn.Linear(hidden_size, 1)
        initialize_weights(self.state_value,init_type)
        

    def forward(self, x):
        x = F.relu(self.linear_in(x))
        value = self.state_value(x)
        return value
