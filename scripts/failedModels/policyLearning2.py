from os.path import abspath, dirname, join

import gym
import numpy as np
import pandas as pd


from torch import nn
import torch
from torch import optim
from torch import distributions
from rye_flex_env.env import RyeFlexEnv
from rye_flex_env.plotter import RyeFlexEnvEpisodePlotter

#Hyperparameters
learning_rate = 0.001
gamma = 0.99
class Policy(nn.Module):
    def __init__(self, state_space, action_space):
        super(Policy, self).__init__()
        self.state_space = state_space
        self.action_space = 2*action_space
        
        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)

        
        
        self.gamma = gamma
        
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
                nn.Tanh()
            )
            return 40*model(x)


def select_action(policy, state):
    #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state.requires_grad = True
    state = policy(state)

    print("State: ", state)
    c = distributions.Normal(state[::2], state[1::2])
    action = c.sample()
    print("Action: ", action)
    print("Std: ", c.scale)
    print("Mean: ", c.loc)
    
    # Add log probability of our chosen action to our history    
    if policy.policy_history.dim() != 0:
        prob = torch.sum(c.log_prob(action), dim=0, keepdim = True)
        policy.policy_history = torch.cat([policy.policy_history, prob])
    else:
        policy.policy_history = (c.log_prob(action))
    
    return action

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
    #print("Policy:", policy.policy_history)
    #print("Rewards: ",rewards)
    loss = (torch.sum(torch.mul(policy.policy_history, rewards).mul(-1), -1))
    
    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
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
    agent = Policy(env.observation_space.shape[0], env.action_space.shape[0])
    
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    
 
    for i in range(100):
        #plotter = RyeFlexEnvEpisodePlotter()
        info = None
        done = False
        # Initial state
        state = env._state.vector
        while not done:
            #state = torch.from_numpy(state).type(torch.FloatTensor)
            action = select_action(agent, state)
            state, reward, done, info = env.step(action.detach().numpy())
            agent.reward_episode.append(-reward)
         #   plotter.update(info)
        update_policy(agent, optimizer)
        print(f"Your score is: {info['cumulative_reward']} NOK")
        #plotter.plot_episode()
    

main()