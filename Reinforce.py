# Importing the Libraries

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from AI import Brain


lr = 5e-4 # 0.0005
miniBatchSize = 100
gamma = 0.99
replay_buffer_size = int(1e5) # memory of the AI
interpolation_parameter = 1e-3 # tou



# Implementing the Experience Replay

class ReplayMemory(object):

    def __init__(self, capacity):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity # Maximum size of the Memory Buffer

        self.memory = []


    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    
    def sample(self, batchSize):
        experiences = random.sample(self.memory, k = batchSize)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return states, next_states, actions, rewards, dones

# Implementing the Deep Q Learning Class

class DQN():

    def __init__(self, stateSize, actionSize):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.stateSize = stateSize
        self.actionSize = actionSize

        self.localQNetwork = Brain(stateSize, actionSize).to(self.device)
        self.targetQNetwork = Brain(stateSize, actionSize).to(self.device)

        self.optimizer = optim.Adam(self.localQNetwork.parameters(), lr = lr)
        
        self.memory = ReplayMemory(replay_buffer_size)
        self.timeStep = 0

    
    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.timeStep = (self.timeStep+1)%4

        if self.timeStep == 0:
            if len(self.memory.memory) > miniBatchSize:
                experiences = self.memory.sample(100)
                self.learn(experiences, gamma)

    def act(self, state, epsilon=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.localQNetwork.eval()
        
        with torch.no_grad():
            actionValues = self.localQNetwork(state)
        
        self.localQNetwork.train()
        
        if random.random() > epsilon:
            return np.argmax(actionValues.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.actionSize))
    
    def learn(self, experiences, gamma):
        states, next_states, actions, rewards, dones = experiences
        nextQTargets = self.targetQNetwork(next_states).detach().max(1)[0].unsqueeze(1)
        
        qTargets = rewards + (gamma*nextQTargets*(1-dones))
        qExpected = self.localQNetwork(states)
        loss = F.mse_loss(qExpected, qTargets)
        
        self.optimizer.zero_grad()
        
        # Back Propagation
        loss.backward()
        
        self.optimizer.step() # Single Optimization step
        self.softUpdate(self.localQNetwork, self.targetQNetwork, interpolation_parameter)
        
    def softUpdate(self, localModel, targetModel, tou):
        
        for targetParam, localParam in zip(localModel.parameters(), localModel.parameters()):
            targetParam.data.copy_(tou * localParam.data + (1.0 - tou) * targetParam.data)
        
    

