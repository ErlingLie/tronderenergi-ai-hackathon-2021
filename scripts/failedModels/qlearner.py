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
learning_rate = 0.1
gamma = 0.99
class Policy(nn.Module):
    def __init__(self, state_space, actions):
        super(Policy, self).__init__()
        self.state_space = state_space
        self.action_space = actions.shape[0]
        self.gamma = gamma
        
        self.l1 = nn.Linear(self.state_space, 32, bias=False)
        self.l2 = nn.Linear(32, self.action_space, bias=False)
        
        self.actions = actions

    def forward(self, state):    
            model = torch.nn.Sequential(
                self.l1,
                nn.Dropout(p=0.6),
                nn.ReLU(),
                self.l2,
                nn.Softmax()
            )
                #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state.requires_grad = True
            logits= model(state)
            #print(mean, std)
            print(logits)
            Q, idx = torch.min(logits, dim=0)
            
            return self.actions[idx], Q


def main() -> None:
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)

    env = RyeFlexEnv(data)
    
    print(env.observation_space, env.action_space)

    
    base_actions = np.array([[1,1], [1,0],[0,1], [1,0.1], [0.1,1]])
    actions = base_actions.copy()
    for i in [-2, -1.5, -1.2, -1, -0.1, -0.01, 0.01, 0.1]:
        actions = np.append(actions, base_actions*i, 0)
    #actions = np.array([[0.1,0], [-0.1,0], [0, 0.1], [0,-0.1]])

    agent = Policy(env.observation_space.shape[0] + 1,actions)
    print(list(agent.parameters()))
    
    lossFunc = nn.MSELoss()
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.1)
    for i in range(30):
        if i == 29:
            plotter = RyeFlexEnvEpisodePlotter()
        env.reset()
        info = None
        done = False
        # Initial state
        state = env._state.vector
        state = np.append([state], [0])
        j = 0
        while not done:
            j += 1
            action, Q1 = agent(state)
            print(action)
            state, reward, done, info = env.step(action)
            state = np.append([state], [j])
            if not done:
                _, Q2 = agent(state)
                loss = lossFunc(Q1, reward+ agent.gamma*Q2)
            else:
                reward =  torch.FloatTensor(np.array(reward))
                reward.requires_grad = True
                loss = lossFunc(Q1,reward)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i == 29:
                plotter.update(info)
        scheduler.step()
        if i % 1 == 0:
            print(f"Your score is: {info['cumulative_reward']} NOK")
    plotter.plot_episode()
    plt.plot(np.arange(len(agent.loss_history)), agent.loss_history)
    plt.show()
    print(list(agent.parameters()))

main()