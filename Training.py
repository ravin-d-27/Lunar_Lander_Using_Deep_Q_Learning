# Importing the Libraries

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from AI import Brain

# Setting up the Lunar Environment

env = gym.make('LunarLander-v2')
stateShape = env.observation_space.shape
print("State Shape: ",stateShape)

stateSize = env.observation_space.shape[0]
print("State Size", stateSize)

numberActions = env.action_space.n
print("Number of Actions: ", numberActions)


# Initializing the Hyper Parameters

lr = 5e-4 # 0.0005
miniBatchSize = 100
gamma = 0.99
replay_buffer_size = int(1e5) # memory of the AI
interpolation_parameter = 1e-3 # tou


# Implementing the Experience Replay

class ReplayMemory(object):

    def __init__(self, capacity):
        
        self.device = torch.cuda("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity # Maximum size of the Memory Buffer

        self.memory = []


    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    
    def sample(self, batchSize):
        experiences = random.sample(self.memory, k = batchSize)

        states = np.vstack([e[0] for e in experiences if e is not None])
        # Converting the states into Pytorch tensors
        states = torch.from_numpy(states).float().to(self.device) # For the designated computed device
        

        actions = np.vstack([e[1] for e in experiences if e is not None])
        # Converting the actions into Pytorch tensors
        actions = torch.from_numpy(actions).long().to(self.device) # For the designated computed device
        
        rewards = np.vstack([e[2] for e in experiences if e is not None])
        # Converting the actions into Pytorch tensors
        rewards = torch.from_numpy(rewards).float().to(self.device) # For the designated computed device


        next_states = np.vstack([e[3] for e in experiences if e is not None])
        # Converting the states into Pytorch tensors
        next_states = torch.from_numpy(next_states).float().to(self.device) # For the designated computed device

        done = np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)
        # Converting the states into Pytorch tensors
        done = torch.from_numpy(done).float().to(self.device) # For the designated computed device


        return states, next_states, actions, rewards, done
        

# Implementing the Deep Q Learning Class

class DQN():

    def __init__(self, stateSize, actionSize):
        self.device = torch.cuda("cuda:0" if torch.cuda.is_available() else "cpu")
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
            return random.choice(np.arrange(self.actionSize))
    
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
        
        