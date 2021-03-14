
import numpy as np
import scipy.io

def get_predicted_consumption(x):
#     A = np.array([1.0000,   -0.7040,    0.1498,   -0.1105,   -0.0393,-0.0383,   -0.0450,   -0.0014,   -0.0232,   0.0116,   -0.0554,   -0.0450,   -0.0887]
    assert(len(x) == 48), f"x is wrong length, got {x.shape}"
    A = scipy.io.loadmat("A.mat")["A"]
    y = -A[:,1:]@x
    return y

def get_predicted_wind_power(wind_speed):
    power_table = np.array([0,0,0,0,3.5,15,33,55,82,115,150,180,208,218,224,225,225,225,225,225,225,225,225,225,225,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

    return power_table[int(wind_speed)]