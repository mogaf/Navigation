[//]: # (Image References)
[image2]: episodes.png "Score"
### Learning Algorithm

Value Based reinforcement learning methods

RL is a branch of machine learning where the agent outputs an action and the environment returns an observation and a reward. The goal of the system is to determine what's the best action to take so that it maximizes the total reward. Deep RL is using NN as a nonlinear function approximator  to calculate the value of actions based directly on observations from the environment. We use Deep Learning to find the optimum params for these approximators. This translates into knowing what's the optimum action to take given that we are in certain state so that we maximize the total reward collected by our agent.

Our neural net is defined as follows : the input layer has the size of the state space , which is 37, then we have two hidden layers of 64 neurons and the output is the size of the possible actions , which in this case is 4. The activation function used by the neurons is ReLU (for more info on activation functions check http://cs231n.github.io/neural-networks-1/#actfun ).

The following code defines the model (can check the full code in [model.py](model.py) :

```python
self.fc1 = nn.Linear(state_size, fc1_units)
self.fc2 = nn.Linear(fc1_units, fc2_units)
self.fc3 = nn.Linear(fc2_units, action_size)
```
and the forward pass :
```python
def forward(self, state):
    """Build a network that maps state -> action values."""
    x = F.relu(self.fc1(state))
    x = F.relu(self.fc2(x))
    return self.fc3(x)
```

The agent is in the file names [dqn_agent.py](dqn_agent.py)
We run the training agent until it gets an average score for the last 100 episodes larger than 13. We run each episode for a number of max_t steps or until it ends, whichever comes first. For every time step, we ask the agent for the action to take given the state we are in, we take that action and we end up with a reward (which can be negative) and a new state. We update the agent with this information and ,if not done, we continue until either done or we run out of time steps. This sequence is in the file [Navigation.ipynd](Navigation.ipynb) and the code is the following:
```python
action = agent.act(state,eps)
env_info = env.step(action)[brain_name]        # send the action to the environment
next_state = env_info.vector_observations[0]   # get the next state
reward = env_info.rewards[0]                   # get the reward
done = env_info.local_done[0]                  # see if episode has finished

```
The agent uses an Epsilon Greedy policy for action selection , meaning that it choses either a random action or the best action with and epsilon probability. The epsilon decreases with the number of full episodes, thus favoring exploitation of the learned model over exploration. I'm starting with a 50% epsilon and I decrease it by 1 % every full episode.

  The agent takes the state, action , reward ,next_state and if the game finished and saves them for future learning. Once enough experiences have accumulated, the agent randomly samples from then and trains the neural network described above. This act of sampling is called Experience Replay. We could've trained the neural network using all the previous steps , but the sequence of exerience tupples is highly correlated and we risk getting swayed by the effects of this correlation. By randomly selecting episodes, we break the correlation. I've used a buffer size of 100k in order to store a large enough number of steps for the agent to remember properly, and a batch size of 128 steps for every training step.

  This simple algorithm is able to train the agent pretty quickly (236 episodes in my case).
  The following graph represents the score of the agent over episodes :
![Score][image2]
The algorithm could be further improved by ,for example, changing the sampling of the steps used for learning. Currenly we sample uniformly from the Replay Buffer, but we could do prioritized experince replay, which means that we give higher weight more meaningful transitions, thus sampling them more frequently.
