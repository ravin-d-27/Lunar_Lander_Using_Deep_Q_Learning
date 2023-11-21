import gymnasium as gym
from collections import deque, namedtuple
import numpy as np
import torch
from Reinforce import DQN


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



### Initialize the DQN Agent

agent = DQN(stateSize, numberActions)


### Training the DQN

numberEpisodes = 2000
maxNumberTimestepsPerEpisode = 1000
epsilonStartingValue = 1.0
epsilonEndingValue = 0.01
epsilonDecayValue = 0.995
epsilon = epsilonStartingValue

ScoresOn100Episodes = deque(maxlen=100)


for episode in range(1, numberEpisodes + 1):
  state, _ = env.reset()
  score = 0
  for t in range(maxNumberTimestepsPerEpisode):
    action = agent.act(state, epsilon)
    next_state, reward, done, _, _ = env.step(action)
    agent.step(state, action, reward, next_state, done)
    state = next_state
    score += reward
    if done:
      break
  ScoresOn100Episodes.append(score)
  epsilon = max(epsilonEndingValue, epsilonDecayValue * epsilon)
  print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(ScoresOn100Episodes)), end = "")
  if episode % 100 == 0:
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(ScoresOn100Episodes)))
  if np.mean(ScoresOn100Episodes) >= 200.0:
    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100, np.mean(ScoresOn100Episodes)))
    torch.save(agent.localQNetwork.state_dict(), 'checkpoint.pth')
    break



import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gym.wrappers.monitoring.video_recorder import VideoRecorder

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action.item())
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'LunarLander-v2')

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()