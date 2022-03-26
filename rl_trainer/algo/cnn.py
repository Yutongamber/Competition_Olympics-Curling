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
    def __init__(self, state_space,action_space, init_type):
        super(Actor, self).__init__()
        self.conv1 = nn.Sequential(     # In: 1*2*30*30
            nn.Conv2d(2, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )                            # Out: 1*16*15*15
        initialize_weights(self.conv1,init_type)
        self.conv2 = nn.Sequential(     # In: 1*16*15*15
            nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(5, stride=2),        # 1*32*6*6
        )
        initialize_weights(self.conv2,init_type)
        self.decision_fc = nn.Linear(1152, action_space)
        initialize_weights(self.decision_fc,init_type)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 1152)
        action_prob = F.softmax(self.decision_fc(x), dim=1)
        return action_prob

class Critic(nn.Module):
    def __init__(self, state_space,init_type):
        super(Critic, self).__init__()
        self.conv1 = nn.Sequential(     # In: 1*2*30*30
            nn.Conv2d(2, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )                            # Out: 1*16*15*15
        initialize_weights(self.conv1,init_type)
        self.conv2 = nn.Sequential(     # In: 1*16*15*15
            nn.Conv2d(16, 32, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(5, stride=2),        # 1*32*5*5
        )
        initialize_weights(self.conv2,init_type)
        self.value = nn.Linear(1152, 1)
        initialize_weights(self.value,init_type)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 1152)
        action_value=self.value(x)
        return action_value
