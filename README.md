

<body>

  <h1>Deep Q-Learning for Lunar Lander</h1>

  <p>This repository contains an implementation of Deep Q-Learning (DQN) to solve the Lunar Lander environment using PyTorch and OpenAI Gym.</p>
  <p>Deep Q-Learning (DQN) is a type of reinforcement learning (RL) algorithm. Reinforcement learning is a subfield of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives feedback in the form of rewards or penalties based on the actions it takes in the environment. The goal of the agent is to learn a policy, a mapping from states to actions, that maximizes the cumulative reward over time.</p>

  <h2>Overview</h2>

  <ol>
    <li>
      <strong>Neural Network Architecture (`Network` class):</strong>
      <ul>
        <li>The neural network is implemented using PyTorch and consists of three fully connected layers.</li>
        <li>The input size is determined by the `state_size` parameter, and the output size is equal to the number of possible actions (`action_size`).</li>
        <li>Rectified Linear Unit (ReLU) activation functions are used between layers to introduce non-linearity.</li>
      </ul>
    </li>
    <li>
      <strong>Agent (`Agent` class):</strong>
      <ul>
        <li>The DQN agent interacts with the environment, collecting experiences and updating the Q-network.</li>
        <li>The agent uses experience replay to store and sample experiences, which helps break correlations in the training data.</li>
        <li>Target networks are used to stabilize the training process.</li>
        <li>The `soft_update` method is employed to perform a soft update of the target Q-network parameters.</li>
        <li>The agent's behavior is epsilon-greedy, where it selects a random action with probability epsilon to encourage exploration.</li>
      </ul>
    </li>
  </ol>

  <h2>Dependencies</h2>

  <ul>
    <li>Python 3.x</li>
    <li>PyTorch</li>
    <li>NumPy</li>
    <li>OpenAI Gym</li>
    <li>gymnasium (Lunar Lander environment)</li><br>
  </ul>

  <p><b>Note: This gymnasium works properly under Linux Environment. If you are in Windows Machine, consider using WSL.</b></p>

  <h2>Instructions</h2>

  <ol>
    <li>Install the required dependencies:
      <pre>pip install -r requirements.txt</pre>
    </li>
    <li>Run the <code>train.py</code> script to train the DQN agent:
      <pre>python3 AI_Runner.py</pre>
    </li>
  </ol>

  <h2>Results</h2>

  <p>The agent's performance is tracked over episodes, and training stops when the environment is considered solved (average score &gt;= 200 over the last 100 episodes).</p>

  <p>The training progress and average scores are printed to the console during training. Additionally, a video of the trained agent playing the game is saved as <code>video.mp4</code> for visual inspection.</p>

<p>Video Reference: </p>
  <a href="https://github.com/ravin-d-27/Lunar_Lander_Using_Deep_Q_Learning/blob/main/Model/video.mp4">Lunar Lander Video</a>


</body>
