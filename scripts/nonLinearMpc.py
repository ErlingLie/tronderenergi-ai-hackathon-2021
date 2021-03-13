import numpy as np
from scipy import optimize
from scipy import linalg

from mpc import *




def MPC_step(N, x0, PV, W, C, spot):

    Aeq, Beq = genFullMatrix(N, x0, PV, W, C)


    Aiq, Biq = genFullIQMatrix(N)

    Q = genQ(N, spot)

    #grad = lambda x: (2*x.reshape(-1,1).T@Q).reshape(-1)

    constraint = lambda x : Aeq@x - Beq
    constraint2 = lambda x: Biq - Aiq@x 
    cons = [{'type':'eq', 'fun': constraint}, {'type':'ineq', 'fun' : constraint2}]
    func = lambda x: (x.reshape(-1,1).T@Q@x.reshape(-1,1)).reshape(-1) + 49*np.max(x.reshape([-1,7])[:,2])
    Ceq = optimize.LinearConstraint(Aeq, Beq, Beq)
    Ciq = optimize.LinearConstraint(Aiq, -np.ones(Aiq.shape[0])*np.inf, Biq)
    x0 = np.concatenate([x0, [0,0,0,0]])
    x0 = np.kron(np.ones(N), x0)

    # x = optimize.minimize(func, x0, method="trust-constr" ,constraints=[Ceq, Ciq])
    x = optimize.minimize(func, x0, method="SLSQP" ,constraints=cons)
    val = x.x[3:7]
    return np.array((val[0]-val[1], val[2]-val[3]))

