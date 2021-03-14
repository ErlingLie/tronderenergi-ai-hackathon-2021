"""
The test script that will be used for evaluation of the "agents" performance
"""
from datetime import datetime
from os.path import abspath, dirname, join

import gym
import numpy as np
import pandas as pd

from rye_flex_env.env import RyeFlexEnv
from rye_flex_env.plotter import RyeFlexEnvEpisodePlotter
from rye_flex_env.states import State, Action

from nonLinearMpc import MPC_step
from predictor import *

import time
class SimpleStateBasedAgent:
    """
    An example agent which always returns a constant action
    """

    def get_action(self, state_vector: np.ndarray) -> np.ndarray:
        """
        Normally one would take the state as input, and select action based on this.
        Since we are taking random action here, knowing the stat is not necessary.
        """

        # Convert from numpy array to State:
        state = State.from_vector(state_vector)

        # Create a state for total production:
        total_production = state.pv_production + state.wind_production

        if total_production > 30:
            # Charging battery with 10 kWh/h and hydrogen with 0 kWh/h
            action = Action(charge_battery=10, charge_hydrogen=0)
            return action.vector
        else:
            # Charging battery with 0 kWh/h and hydrogen with 10 kWh/h
            return np.array([0, 10])


def main() -> None:
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/test.csv"), index_col=0, parse_dates=True)

    env = RyeFlexEnv(data=data)
    plotter = RyeFlexEnvEpisodePlotter()
    data2 = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)
    data = pd.concat([data2, data])
    # Reset episode to feb 2021, and get initial state
    state = env.reset(start_time=datetime(2021, 2, 1, 0, 0))

    # INSERT YOUR OWN ALGORITHM HERE
    agent = SimpleStateBasedAgent()

    info = {}
    done = False
    N = 20
    while not done:
        t0 = time.time()
        PV = data.loc[env._time:env._time + N*env._time_resolution, "pv_production"]
        W = data.loc[env._time:env._time + N*env._time_resolution, "wind_production"]
        C = data.loc[env._time:env._time + N*env._time_resolution, "consumption"]
        spot = data.loc[env._time:env._time + N*env._time_resolution, "spot_market_price"]
        #print("State t: ", state[0] - state[1] - state[2] + action[0] + action[1])

        # C = data.loc[env._time - 47*env._time_resolution:env._time, "consumption"]
        # Wind = data.loc[env._time:env._time + N*env._time_resolution, "wind_speed_50m:ms"]
        # C_estim = [np.array(C[-1])]
        # for i in range(N):
        #     c = get_predicted_consumption(C[-48:])
        #     C_estim.append(c)
        #     C = np.concatenate([C, c])
        # W = []
        # for x in Wind:
        #     W.append(get_predicted_wind_power(x))
        # W = np.array(W)
        # C = np.hstack(C_estim)
        #INSERT YOUR OWN ALGORITHM HERE
        action = MPC_step(N, state[3:6],PV[1:], W[1:], C[1:], spot[1:] )
        # action = np.array([0,0])
        print(env._time)
        print(time.time() - t0)
        state, reward, done, info = env.step(action)

        plotter.update(info)

    print(f"Your test score is: {info['cumulative_reward']} NOK")

    plotter.plot_episode()


if __name__ == "__main__":
    main()
