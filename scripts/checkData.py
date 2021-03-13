from matplotlib import pyplot as plt

from datetime import datetime, timedelta
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd

from rye_flex_env.env import RyeFlexEnv
from rye_flex_env.plotter import RyeFlexEnvEpisodePlotter


def main() -> None:
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)

    time = data.index.min()
    
    time_delta = timedelta(hours = 1)
    timeMax = time + 24*time_delta
    print(time.hour)

    # plt.subplot(5,1,1)
    # plt.plot(data.loc[time :timeMax, "wind_speed_2m:ms"])
    # plt.subplot(5,1,2)
    # plt.plot(data.loc[time :timeMax,"wind_speed_10m:ms"])
    # plt.subplot(5,1,3)
    # plt.plot(data.loc[time :timeMax, "wind_speed_50m:ms"])
    # plt.subplot(5,1,4)
    # plt.plot(data.loc[time :timeMax, "wind_speed_100m:ms"])
    # plt.subplot(5,1,5)
    # plt.plot(data.loc[time :timeMax, "wind_production"])
    # plt.show()


    # plt.subplot(5,1,1)
    # plt.plot(data.loc[time :timeMax, "wind_dir_2m:d"])
    # plt.subplot(5,1,2)
    # plt.plot(data.loc[time :timeMax,"wind_dir_10m:d"])
    # plt.subplot(5,1,3)
    # plt.plot(data.loc[time :timeMax, "wind_dir_50m:d"])
    # plt.subplot(5,1,4)
    # plt.plot(data.loc[time :timeMax, "wind_dir_100m:d"])
    # plt.subplot(5,1,5)
    # plt.plot(data.loc[time :timeMax, "wind_production"])
    # plt.show()


    # plt.subplot(5,1,1)
    # plt.plot(np.where(data.loc[time :timeMax, "wind_speed_2m:ms"]<3.5, 1,0) )
    # plt.subplot(5,1,2)
    # plt.plot(np.where(data.loc[time :timeMax, "wind_speed_10m:ms"]<3.5, 1,0))
    # plt.subplot(5,1,3)
    # plt.plot(np.where(data.loc[time :timeMax, "wind_speed_50m:ms"]<3.5, 1,0))
    # plt.subplot(5,1,4)
    # plt.plot(np.where(data.loc[time :timeMax, "wind_speed_100m:ms"]<3.5, 1,0))
    # plt.subplot(5,1,5)
    # plt.plot(data.loc[time :timeMax, "wind_production"])
    # plt.show()

    # w1 = np.array(data.loc[time :timeMax, "wind_speed_2m:ms"])
    # w2 = data.loc[time :timeMax, "wind_speed_10m:ms"]
    # w3 = data.loc[time :timeMax, "wind_speed_50m:ms"]
    # w4 = data.loc[time :timeMax, "wind_speed_100m:ms"]
    # P = np.array(data.loc[time :timeMax, "wind_production"])
    # print(np.correlate(w1, P, "full"))
    # plt.subplot(5,1,1)
    # plt.plot(np.correlate(w1,P, "full" )) 
    # plt.subplot(5,1,2)
    # plt.plot(np.correlate(w2,P ,"full"))
    # plt.subplot(5,1,3)
    # plt.plot(np.correlate(w3,P ,"full"))
    # plt.subplot(5,1,4)
    # plt.plot(np.correlate(w4,P , "full"))
    # plt.subplot(5,1,5)
    # plt.plot(np.correlate(P,P, "full" ))
    # plt.show()

    plt.plot(data.loc[time:timeMax, "consumption"])
    plt.show()
if __name__ == "__main__":
    main()
