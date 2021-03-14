from os.path import abspath, dirname, join

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from torch import nn
import torch
from torch import optim
from torch import distributions


class LinearClassifier(nn.Module):
    def __init__(self, N):
        super(LinearClassifier, self).__init__()

        self.l1 = nn.Linear(N, 128)
        self.l2 = nn.Linear(128, 1)
        self.ReLU = nn.ReLU()
    def forward(self, x):
        x = self.l1(x)
        x = self.ReLU(x)
        x = self.l2(x)
        x = self.ReLU(x)
        return x

def main() -> None:
    root_dir = dirname(abspath(join(__file__, "../")))
    data = pd.read_csv(join(root_dir, "data/train.csv"), index_col=0, parse_dates=True)

    time = data.index.min()
    timeMax = data.index.max()

    w1 = np.array(data.loc[time :timeMax, "wind_speed_2m:ms"])
    w2 = data.loc[time :timeMax, "wind_speed_10m:ms"]
    w3 = data.loc[time :timeMax, "wind_speed_50m:ms"]
    w4 = data.loc[time :timeMax, "wind_speed_100m:ms"]
    P = np.array(data.loc[time :timeMax, "wind_production"])
    N = 20
    model = LinearClassifier(N*2)
    lf = torch.nn.MSELoss()

    optimizer = optim.Adam(model.parameters(),0.01)
    for i in range(100):
        for j in range(0, len(w3)-50, 10):
            x = []
            target = []
            for ii in range(10):
                x.append(np.concatenate([w3[j+ii +1:j+ii+21], P[j+ii:j+ii+20]]))
                target.append(P[j+ii+20])
            x = np.vstack(x)
            target = np.array(target).reshape((-1,1))
            target = torch.tensor(target).type(torch.float)
            x = torch.tensor(x).type(torch.float)
            y = model(x)
            loss = lf(y, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(loss.item())



main()