"""
The test script that will be used for evaluation of the "agents" performance
"""
from datetime import datetime
from os.path import abspath, dirname, join

import gym
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from matplotlib import pyplot as plt

from rye_flex_env.env import RyeFlexEnv
from rye_flex_env.plotter import RyeFlexEnvEpisodePlotter
from rye_flex_env.states import State


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    def __init__(self, state):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=11, dim_u = 4, dim_z=8)
        self.kf.F = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [1, -1, -1, 0, 0, 0, 0, 0, 0 ,0 ,0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ,0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ,0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,1]])

        self.kf.B = np.zeros([11,4])
        self.kf.B[3,0] = 0.85
        self.kf.B[3,1] = -1
        self.kf.B[4,2] = 0.325
        self.kf.B[4,3] = -1
        self.kf.B[5,1] = -1
        self.kf.B[5,3] = -1
        print(self.kf.B)
        self.kf.H = np.zeros((8, 11))
        self.kf.H[:8,:8] = np.eye(8)

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[11:, 11:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        #print(self.kf.x.shape)
        self.kf.x[:8] = state.reshape(-1,1)
        #print(self.kf.x.shape)

    def update(self, states):
        """
        Updates the state vector with observed bbox.
        """
        self.kf.update(states)
        #print("Update: ", self.kf.x.shape)

    def predict(self, u):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        #print(u)
        u = np.clip(u, [0, 0, 0, 0], [400, 400, 55, 100])
        #print(u)
        self.kf.predict(u.reshape([4,1]))
        #print("Predict: ", self.kf.x.shape)
        if self.kf.x[3] > 500:
            self.kf.x[3] = 500
        elif self.kf.x[3] < 0:
            self.kf.x[3] = 0
        if self.kf.x[4] > 1670:
            self.kf.x[4] = 1670
        elif self.kf.x[4] < 0:
            self.kf.x[4] = 0

        return self.kf.x[:8]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x



class RandomActionAgent:
    def __init__(self, action_space: gym.spaces.Box):
        self._action_space = action_space
        self.state_history = []
        self.predicted_states = np.zeros(8)
        self.predicted_states2 = np.zeros(8)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Normally one would take state as input, and select action based on this.
        Since we are taking random action here, knowing the stat is not necessary.
        """
        # print("Step error1: ", np.linalg.norm((state-self.predicted_states)[:3]))
        # print("Step error2: ", np.linalg.norm((state-self.predicted_states2)[:3]))
        # #print("States: ", state[:3], self.state_history[-1][:3], self.state_history[-2][:3], predicted2[:3], np.linalg.norm(predicted2[:3]))
        
        # da = 11/6*state - 3*self.state_history[-1] + 3/2*self.state_history[-2] - 1/3*self.state_history[-3]
        # dda = state - 3*self.state_history[-1] + 3*self.state_history[-2] - self.state_history[-3]
        # da2 = state - self.state_history[-1]
        # print("Actual error: ", np.linalg.norm(da2[:3]))
        # #print("Predicted error2: ", np.linalg.norm(da2[:3]))
        self.state_history.append(state.reshape([8,1]))

        # self.predicted_states = state + (da + 1/2*dda)*np.array([1,1,1,0,0,0,0,0])
        # self.predicted_states2 = state + da2*np.array([1,1,1,0,0,0,0,0])
        states = State.from_vector(state)

        surplus = (states.pv_production + states.wind_production - states.consumption)
        action = np.zeros(2)
        if surplus > 0:
            battery = np.min([500.0 - states.battery_storage, surplus])
            battery = np.min([battery, 400])
            hydro = surplus - battery
            return np.array([battery, hydro])
        else:
            battery = np.max([-states.battery_storage, surplus])*1*(states.battery_storage/500)**2
            battery = np.max([-400, battery])
            hydro = np.max([-states.hydrogen_storage, surplus - battery])*1*(states.hydrogen_storage/800)**2
            return np.array([battery, hydro])

class KalmanAgent:
    def __init__(self, action_space: gym.spaces.Box, state):
        self._action_space = action_space
        self.state_history = []
        self.KF = KalmanBoxTracker(state)
        self.prev_u = np.zeros(2)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Normally one would take state as input, and select action based on this.
        Since we are taking random action here, knowing the stat is not necessary.
        """
        self.KF.update(state)
        self.state_history.append(self.KF.kf.x)
        u = np.zeros(4)
        if self.prev_u[0] < 0:
            u[1] = -self.prev_u[0]
        else:
            u[0] = self.prev_u[0]
        if self.prev_u[1] < 0:
            u[3] = -self.prev_u[1]
        else:
            u[2] = self.prev_u[1]
        self.predicted_states = self.KF.predict(state)
        states = State.from_vector(self.predicted_states)

        surplus = states.pv_production + states.wind_production - states.consumption
        if surplus > 0:
            battery = np.min([500.0 - states.battery_storage, surplus])
            
            # print("Consumption: ", states.consumption)
            # print("Wind: ", states.wind_production)
            # print("PV: ", states.pv_production)
            # print("Surplus: ", surplus)
            # print("Battery: ", battery)
            # print("Hydro: ", hydro)
            hydro = surplus - battery
            self.prev_u = np.array([battery, hydro])
            #print(states.pv_production + states.wind_production - states.consumption - ac)
            return self.prev_u
        else:
            battery = np.max([-states.battery_storage, surplus])
            hydro = np.max([-states.hydrogen_storage, surplus - battery])
            # if hydro < 0:
            #     print("Consumption: ", states.consumption)
            #     print("Wind: ", states.wind_production)
            #     print("PV: ", states.pv_production)
            #     print("Surplus: ", surplus)
            #     print("Battery: ", battery)
            #     print("Hydro: ", hydro)
            self.prev_u = np.array([battery, hydro])
            return self.prev_u

def main() -> None:
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data=data)
    env.reset(start_time=datetime(2020, 2, 1, 0, 0))
    plotter = RyeFlexEnvEpisodePlotter()

    # INSERT YOUR OWN ALGORITHM HERE
    #agent = KalmanAgent(env.action_space)

    # Example with random initial state
    info = {}
    done = False
    # Initial state
    state = env._state.vector
    agent = RandomActionAgent(env.action_space)
    #agent = KalmanAgent(env.action_space, state)
    i = 0
    while not done:

        # INSERT YOUR OWN ALGORITHM HERE
        #print(state[0], data.at[env._time + env._time_resolution, "consumption"])
        state[0] = data.at[env._time + env._time_resolution, "consumption"]
        state[1] = data.at[env._time + env._time_resolution, "pv_production"]
        state[2] = data.at[env._time + env._time_resolution, "wind_production"]
        action = agent.get_action(state)
        
        #action = np.array([0,0])
        #print("State t: ", state[0] - state[1] - state[2] + action[0] + action[1])

        state, reward, done, info = env.step(action)

        #print("State t+1: ", state[0] - state[1] - state[2] + action[0] + action[1])

        plotter.update(info)
        i += 1
        # if i == 24*10:
        #     break

    print(f"Your test score is: {info['cumulative_reward']} NOK")
    plotter.plot_episode(True)
    # states = []
    # for i, a in zip(plotter._states, plotter._actions):
    #     states.append([i["consumption"], -i["pv_production"], -i["wind_production"], a["charge_battery"], a["charge_hydrogen"],-i["grid_import"]])
    # states = np.vstack(states)
    # plt.subplot(2,1,1)
    # plt.plot(np.sum(states, 1))
    # plt.subplot(2,1,2)
    # plt.plot(np.sum(np.hstack([states[0:-1,:3], states[1:,3:]]), 1))
    # plt.show()

    # t = np.arange(len(agent.state_history))
    # agent.state_history = np.hstack(agent.state_history)
    # plt.figure()
    # plt.subplot(8,1,1)
    # plt.plot(t, agent.state_history[0,:])
    # plt.subplot(8,1,2)
    # plt.plot(t, agent.state_history[1,:])
    # plt.subplot(8,1,3)
    # plt.plot(t, agent.state_history[2,:])
    # plt.subplot(8,1,4)
    # plt.plot(t, agent.state_history[3,:])
    # plt.subplot(8,1,5)
    # plt.plot(t, agent.state_history[4, :])
    # plt.subplot(8,1,6)
    # plt.plot(t, agent.state_history[5, :])
    # plt.subplot(8,1,7)
    # plt.plot(t, agent.state_history[6, :])
    # plt.subplot(8,1,8)
    # plt.plot(t, agent.state_history[7, :])

    # plt.show()

if __name__ == "__main__":
    main()
