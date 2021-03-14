from os.path import abspath, dirname, join

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from torch import nn
import torch
from torch import optim
from torch import distributions
from rye_flex_env.env import RyeFlexEnv
from rye_flex_env.plotter import RyeFlexEnvEpisodePlotter

#Hyperparameters
learning_rate = 0.01
gamma = 0.99
class Policy(nn.Module):
    def __init__(self, state_space, action_space):
        super(Policy, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        
        self.l1 = nn.Linear(self.state_space, 256, bias=False)
        self.l2 = nn.Linear(256, self.action_space, bias=False)
        
        # Episode policy and reward history 
        self.policy_history = torch.Tensor() 
        self.policy_history.requires_grad = True
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []
    def forward(self, x):    
            model = torch.nn.Sequential(
                self.l1,
                nn.Dropout(p=0.6),
                nn.ReLU(),
                self.l2,
                nn.Softmax()
            )
            return model(x)


def select_action(policy, state, actions):
    #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state.requires_grad = True
    logits= policy(state)
    #print(mean, std)
    c = distributions.Categorical(logits)
    action = c.sample().reshape(1,1)
    
    # Add log probability of our chosen action to our history    
    if policy.policy_history.dim() != 0:
        prob = c.log_prob(action).flatten()
        policy.policy_history = torch.cat([policy.policy_history, prob])
    else:
        policy.policy_history = (c.log_prob(action))
    

    return actions[action]

def update_policy(policy, optimizer):
    R = 0
    rewards = []
    
    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0,R)
        
    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards.requires_grad = True
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    
    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, rewards).mul(-1), -1))
    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
    #Save and intialize episode history counters
    policy.loss_history.append(loss.data.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = torch.Tensor()
    policy.policy_history.requires_grad = True
    policy.reward_episode= []

def main() -> None:
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)

    env = RyeFlexEnv(data)
    
    print(env.observation_space, env.action_space)

    
    base_actions = np.array([[1,1], [1,0],[0,1], [1,0.1], [0.1,1]])
    actions = base_actions.copy()
    for i in [-10, -5, -1, 5, 10]:
        actions = np.append(actions, base_actions*i, 0)

    agent = Policy(env.observation_space.shape[0], actions.shape[0])
    print(list(agent.parameters()))
    
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.1)
    for i in range(100):
        #plotter = RyeFlexEnvEpisodePlotter()
        env.reset()
        info = None
        done = False
        # Initial state
        state = env._state.vector
        while not done:
            action = select_action(agent, state,actions)
            state, reward, done, info = env.step(action)
            agent.reward_episode.append(-reward)
         #   plotter.update(info)
        update_policy(agent, optimizer)
        scheduler.step()
        if i % 1 == 0:
            print(f"Your score is: {info['cumulative_reward']} NOK")
        #plotter.plot_episode()
    plt.plot(np.arange(len(agent.loss_history)), agent.loss_history)
    plt.show()
    print(list(agent.parameters()))

main()