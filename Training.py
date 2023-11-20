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