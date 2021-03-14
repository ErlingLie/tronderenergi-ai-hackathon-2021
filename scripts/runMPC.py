"""
The test script that will be used for evaluation of the "agents" performance
"""
from datetime import datetime
from os.path import abspath, dirname, join

import gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from rye_flex_env.env import RyeFlexEnv
from rye_flex_env.plotter import RyeFlexEnvEpisodePlotter
from rye_flex_env.states import State
from predictor import *
from mpc import MPC_step
# from nonLinearMpc import MPC_step

def main() -> None:
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/test.csv"), index_col=0, parse_dates=True)
    env = RyeFlexEnv(data=data)
    env.reset(start_time=datetime(2021, 2, 1, 0, 0))
    data2 = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    data = pd.concat([data2, data])
    plotter = RyeFlexEnvEpisodePlotter()

    # INSERT YOUR OWN ALGORITHM HERE
    #agent = KalmanAgent(env.action_space)

    # Example with random initial state
    info = {}
    done = False
    # Initial state
    state = env._state.vector
    N = 28
    while not done:
        #PV = data.loc[env._time:env._time + N*env._time_resolution, "pv_production"]
        #W = data.loc[env._time:env._time + N*env._time_resolution, "wind_production"]
        #C = data.loc[env._time:env._time + N*env._time_resolution, "consumption"]
        spot = data.loc[env._time:env._time + N*env._time_resolution, "spot_market_price"]
        #print("State t: ", state[0] - state[1] - state[2] + action[0] + action[1])

        C = data.loc[env._time - 47*env._time_resolution:env._time, "consumption"]
        PV = data.loc[env._time - 47*env._time_resolution:env._time, "pv_production"]
        Wind = data.loc[env._time:env._time + N*env._time_resolution, "wind_speed_50m:ms"]
        Wind_prod_last = data.loc[env._time, "wind_production"]
        C_estim = [np.array(C[-1])]
        PV_estim = [np.array(PV[-1])]
        for i in range(N):
            c = get_predicted_consumption(C[-48:])
            C_estim.append(c)
            C = np.concatenate([C, c])
            pv = get_predicted_solar_power(PV[-48:])
            PV_estim.append(pv)
            PV = np.concatenate([PV, pv])
        # W = []
        # for x in Wind:
        #     W.append(get_predicted_wind_power(x))
        # W = np.array(W)
        W = get_predicted_wind_power_stupid(Wind,Wind_prod_last,N)
        C = np.hstack(C_estim)
        action = MPC_step(N, state[3:6],PV[1:], W[1:], C[1:], spot[1:] )
        state, reward, done, info = env.step(action)
        print(env._time)
        plotter.update(info)

    print(f"Your test score is: {info['cumulative_reward']} NOK")
    plotter.plot_episode()


if __name__ == "__main__":
    main()
