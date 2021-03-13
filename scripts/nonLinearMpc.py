import numpy as np
from scipy import optimize
from scipy import linalg

def generateIQMatrics():
    A = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    B = np.array([[0.85, 0, 0, 0], [ 0, 1, 0, 0], [0, 0, 0.325, 0], [0, 0, 0, 1]])
    C = np.zeros((4,7))
    C[:,3:] = -np.eye(4)

    A = np.hstack([A, B])
    A = np.vstack([A, C])

    b = np.array([500, 0, 1670, 0, 0, 0, 0, 0])
    return A, b

def genFullIQMatrix(N):
    A, b = generateIQMatrics()
    matrix = np.kron(np.eye(N), A)
    B = np.kron(np.ones(N), b)
    return matrix, B


def generateEQInitalMatrix(x0):
    A1 = np.array([[1, 0, 0, 0.85, -1, 0, 0], [0, 1, 0, 0, 0, 0.325, -1]])
    A1[:, 3:] *= -1
    b0 = np.array([[1, 0, 0], [0,1,0]])@x0
    return A1, b0
def generateEQMatrices(PV, W, C):
    A1 = np.array([[1, 0, 0, 0.85, -1, 0, 0], [0, 1, 0, 0, 0, 0.325, -1]])
    B1 = np.zeros((2,7))
    B1[:2,:2] = -np.eye(2)
    Aeq1 = np.hstack([A1, B1])
    b1 = np.array([0,0])

    Aeq2 = np.array([[0, 0, -1, 1, -1, 1, -1]])
    b2 = np.array([PV + W - C])

    Aeq2 = np.hstack([Aeq2, np.zeros((1,7))])

    A = np.vstack([Aeq1, Aeq2])
    b = np.hstack([b1, b2])

    return A, b

def genQ(N, spot):
    Q = []
    for i in range(N):
        q = np.zeros((7,7))
        q[2,2] = 0.05 + spot[i]
        Q.append(q)
    return linalg.block_diag(*Q)

def genFullMatrix(N, x0, PV, W, C):
    matrix = np.zeros(((N-1)*3+2, 7*N))
    A1, b0 = generateEQInitalMatrix(x0)
   # print(A.shape, b.shape, A1.shape, b0.shape)
    matrix[:2,:7] = A1
    c = 2
    r = 0
    B = b0
    for i in range(N-1):
        pv = PV[i]
        w = W[i]
        cons = C[i]
        A, b = generateEQMatrices(pv, w, cons)
        matrix[c:c+3, r:r+14] = A
        B = np.concatenate([B, b])
        c += 3
        r += 7
    return matrix, B




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

