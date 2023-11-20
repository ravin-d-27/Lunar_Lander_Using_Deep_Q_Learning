# Importing the Libraries

import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from torch.autograd import Variable
from collections import deque, namedtuple

# Building the Architecture of the Neural Network

class Brain(nn.Module):

    def __init__(self, stateSize, actionSize,seed = 42): # According to Lunar Lander in Gymnasium, stateSize is 8. 
    #Seed is the randomness, ActionSize according to Lunar Lander is 4
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(stateSize, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, actionSize)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        return self.fc3(x)
        

