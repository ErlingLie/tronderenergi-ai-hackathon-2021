from datetime import datetime

import pandas as pd
from statsmodels.tsa import x13

from rye_flex_env.env import RyeFlexEnv
from rye_flex_env.plotter import RyeFlexEnvEpisodePlotter
import scipy.io
import numpy as np
import statsmodels.api as sm


from matplotlib import pyplot as plt

from datetime import datetime, timedelta
from os.path import abspath, dirname, join
def test_episodes():
    """
    Test to check that length of episode,
    cumulative reward and done signal are sent correctly
    """

    data = pd.read_csv("data/train.csv", index_col=0, parse_dates=True)

    env = RyeFlexEnv(data=data)
    plotter = RyeFlexEnvEpisodePlotter()
    length = int(env._episode_length.days * 24)

    # Example with random initial state
    done = False
    cumulative_reward = env._cumulative_reward




    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        new_cumulative_reward = info["cumulative_reward"]

        assert round(new_cumulative_reward - cumulative_reward, 5) == round(reward, 5)

        cumulative_reward = new_cumulative_reward
        plotter.update(info)

    assert len(plotter._states) == length

    plotter.plot_episode(show=True)
    
    mydata = np.array(wind)
    # scipy.io.savemat('wind.mat', mydata)

    # Example where environment are set to partial known state
    env.reset(start_time=datetime(2020, 2, 3), battery_storage=1)

    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        plotter.update(info)

    assert len(plotter._states) == length
    plotter.plot_episode(show=False)



def get_data():
    data = pd.read_csv("data/train.csv", index_col=0, parse_dates=True)

    env = RyeFlexEnv(data=data)

    print(env._measured_wind_production_data.copy)


# def AR_func():
#     data = pd.read_csv("data/train.csv", index_col=0, parse_dates=True)
#     df = pd.DataFrame(index=pd.date_range(start=datetime(2020, 2, 1), end=datetime(2021, 1, 3), freq='h'))
#     df = df.join(data)
#     print(df.consumption)
#     # df.consumption.interpolate(inplace=True)
#     ar_model = sm.tsa.AutoReg(df.consumption, missing='drop',lags=9)
#     ar_res = ar_model.fit(maxlag=9, method='mle', disp=-1)
#     pred = ar_model.predict(start=datetime(2020, 2, 1), end=datetime(2021,1,2))

#     return pred

def get_predicted_consumption(x):
#     A = np.array([1.0000,   -0.7040,    0.1498,   -0.1105,   -0.0393,-0.0383,   -0.0450,   -0.0014,   -0.0232,   0.0116,   -0.0554,   -0.0450,   -0.0887]
    A = scipy.io.loadmat("A.mat")["A"]
    y = -A[:,1:]@x
    return y

def test_ARmodel(model_of):
    data = pd.read_csv("data/test.csv", index_col=0, parse_dates=True)

    time_min = data.index.min()
    time_max = data.index.max()
    env = RyeFlexEnv(data=data)

    c = data.loc[time_min:time_max, model_of]
    estim = []
    y = []
    days = 10
    for i in range(400, 400 +24*days):
        if model_of == "consumption":
            estim.append(get_predicted_consumption(c[i:i+48]))
        if model_of == "pv_production":
            estim.append(get_predicted_solar_power(c[i:i+48]))
        y.append(c[i + 48])
    plt.plot(np.arange(0,24*days), estim, 'r', label="estimated")
    plt.plot(np.arange(0,24*days), y, 'g', label="real")
    plt.legend()
    plt.show()

def get_predicted_wind_power_stupid(wind_speed_vec, last_m,N):
    power_table = np.array([0,0,0,0,3.5,15,33,55,82,115,150,180,208,218,224,225,225,225,225,225,225,225,225,225,225,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    
    this = last_m
    a = [this]
    for i in range(1,N):
        this = this + 5*(wind_speed_vec[i]-wind_speed_vec[i-1])
        a.append(this)

    return np.array(a)

def get_predicted_wind_power(wind_speed):
    power_table = np.array([0,0,0,0,3.5,15,33,55,82,115,150,180,208,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    return power_table[int(wind_speed)]

def get_predicted_solar_power(x):
    assert(len(x) == 48), f"x is wrong length, got {x.shape}"
    A = scipy.io.loadmat("pv_ARmodel.mat")["pv_AR_model"]
    y = -A[:,1:]@x
    return y

def test_predicted_wind_power():
    #root_dir = dirname(abspath(join(__file__, "./")))
    data = pd.read_csv( "data/test.csv", index_col=0, parse_dates=True)
    env = RyeFlexEnv(data=data)
    time = data.index.min()
    
    time_delta = timedelta(days = 1)
    timeMax = time + time_delta*365
    w1 = np.array(data.loc[time :timeMax, "wind_speed_2m:ms"])
    w2 = data.loc[time :timeMax, "wind_speed_10m:ms"]
    w3 = data.loc[time :timeMax, "wind_speed_50m:ms"]
    w4 = data.loc[time :timeMax, "wind_speed_100m:ms"]
    P = np.array(data.loc[time :timeMax, "wind_production"])

    # estim = []
    # for data in w4:
    #     estim.append(get_predicted_wind_power(data))
    Wind_prod_last = data.loc[env._time, "wind_production"]
    estim = get_predicted_wind_power_stupid(w1,Wind_prod_last,28)


    plt.plot(np.arange(w3.shape[0]), w4, label='Wind')
    plt.plot(np.arange(w3.shape[0]), P, label='Power')
    plt.plot(np.arange(w3.shape[0]), estim, label = 'Estimated power')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # test_episodes()
    test_predicted_wind_power()
    # test_ARmodel("pv_production")

