# Importing the Libraries

import gymnasium as gym

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

