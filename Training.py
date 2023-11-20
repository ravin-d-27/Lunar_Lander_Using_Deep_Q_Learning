# Importing the Libraries

import gymnasium as gym
import numpy as np
import torch

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
        