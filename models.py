import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.layer_1 = nn.Linear(state_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer_2 = nn.Linear(128, 128)
        self.layer_3 = nn.Linear(128, action_size)
        self.reset_parameters()
        return
    
    def reset_parameters(self):
        self.layer_1.weight.data.uniform_(*hidden_init(self.layer_1))
        self.layer_2.weight.data.uniform_(*hidden_init(self.layer_2))
        self.layer_3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        x = F.relu(self.bn1(self.layer_1(states)))
        x = F.relu(self.layer_2(x))
        return torch.tanh(self.layer_3(x))

class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.state_fc = nn.Linear(state_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.value_fc1 = nn.Linear(128 +action_size, 128)
        self.output_fc = nn.Linear(128,1)
        self.reset_parameters()
        return
    
    def reset_parameters(self):
        self.state_fc.weight.data.uniform_(*hidden_init(self.state_fc))
        self.value_fc1.weight.data.uniform_(*hidden_init(self.value_fc1))
        self.output_fc.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, action):
        xs = F.leaky_relu(self.bn1(self.state_fc(states)))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.value_fc1(x))
        return self.output_fc(x)

