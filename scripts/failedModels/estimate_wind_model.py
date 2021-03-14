import numpy as np
import pandas as pd

from scipy.optimize import least_squares

from matplotlib import pyplot as plt

from datetime import datetime, timedelta
from os.path import abspath, dirname, join

def estimFunction(a,b, P, V): 
    P_x = P[:-1]
    b = b.reshape([-1,1])
    x1 = V@b
    x2 = a*P_x.reshape([-1,1])

    result = np.where(V[:,2]>3.5, x1+x2, 0)

    return result


def residuals(a, b, P, V):
    P_y = P[1:]
    estim = estimFunction(a,b,P,V)
    return (P_y.reshape([-1,1]) - estim).reshape([-1])



root_dir = dirname(abspath(join(__file__, "../")))
data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)

time = data.index.min()

time_delta = timedelta(hours = 1)
timeMax = time + 30*24*time_delta
w1 = np.array(data.loc[time :timeMax, "wind_speed_2m:ms"])
w2 = data.loc[time :timeMax, "wind_speed_10m:ms"]
w3 = data.loc[time :timeMax, "wind_speed_50m:ms"]
w4 = data.loc[time :timeMax, "wind_speed_100m:ms"]
P = np.array(data.loc[time :timeMax, "wind_production"])

V = np.vstack([w1,w2,w3,w4]).T[1:]
print(P.shape, V.shape)
fun = lambda x: residuals(x[0], x[1:], P, V)

x = least_squares(fun, np.ones(5))
print(np.mean(np.abs(x.fun)))
print(np.mean(np.abs(P[1:] - P[:-1])))