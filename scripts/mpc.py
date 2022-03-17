import numpy as np
from scipy import optimize
from scipy import linalg
from scipy import sparse
def generateIQMatrics():
    A = np.array([[1, 0, 0],
                  [-1, 0, 0],
                  [0, 1, 0],
                  [0, -1, 0]])
    B = np.array([[0.85, 0, 0, 0], [ 0, 1, 0, 0], [0, 0, 0.325, 0], [0, 0, 0, 1]])
    C = np.zeros((4,7))
    C[:,3:] = -np.eye(4)


    A = np.hstack([A, B])
    A = np.vstack([A, C, -C])

    b = np.array([500, 0, 1670, 0, 0, 0, 0, 0, 400, 400, 55, 100])
    return A, b

def genFullIQMatrix(N, PV, W, C):
    A, b = generateIQMatrics()
    matrix = np.kron(np.eye(N), A)
    B = np.kron(np.ones(N), b)
    Aiq2 = np.array([[0, 0, -1, 1, -1, 1, -1]])
    A2 = np.kron(np.eye(N), Aiq2)
    b2 = PV + W-C

    matrix = np.vstack([matrix, A2])
    B = np.hstack([B,b2])

    return matrix, B


def generateEQInitalMatrix(x0):
    A1 = np.array([[1, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0]])
    b0 = np.array([[1, 0, 0], [0,1,0]])@x0
    return A1, b0
def generateEQMatrices():
    A1 = np.array([[1, 0, 0, 0.85, -1, 0, 0],
                   [0, 1, 0, 0, 0, 0.325, -1]])
    B1 = np.zeros((2,7))
    B1[:2,:2] = -np.eye(2)
    Aeq1 = np.hstack([A1, B1])
    b1 = np.array([0,0])

    # Aeq2 = np.array([[0, 0, -1, 1, -1, 1, -1]])
    # b2 = np.array([PV + W - C])

    # Aeq2 = np.hstack([Aeq2, np.zeros((1,7))])

    # A = np.vstack([Aeq1, Aeq2])
    # b = np.hstack([b1, b2])

    return Aeq1, b1

def genQ(N, spot):
    Q = []
    for i in range(N):
        q = np.zeros((7,7))
        q[2,2] = 0.05 + spot[i]
        Q.append(q)
    return linalg.block_diag(*Q)

def genFullMatrix(N, x0):
    matrix = np.zeros(((N-1)*2+2, 7*N))
    A1, b0 = generateEQInitalMatrix(x0)
   # print(A.shape, b.shape, A1.shape, b0.shape)
    matrix[:2,:7] = A1
    c = 2
    r = 0
    B = b0
    for i in range(N-1):
        A, b = generateEQMatrices()
        matrix[c:c+2, r:r+14] = A
        B = np.concatenate([B, b])
        c += 2
        r += 7
    return matrix, B




def MPC_step(N, x0, PV, W, C, spot, peak_power):

    Aeq, Beq = genFullMatrix(N, x0,)


    Aiq, Biq = genFullIQMatrix(N, PV, W, C)

    Q = genQ(N, spot)
   
    c = np.diag(Q)
    c = np.concatenate([c,[49]])
    At = np.kron(np.eye(N),np.array([[0,0,1,0,0,0,0]]))
    At = np.hstack([At, -np.ones((N,1))])
    At = np.vstack([At, np.zeros([1,At.shape[1]])])
    At[-1,-1] = -1
    Aiq = np.hstack([Aiq, np.zeros((Aiq.shape[0],1))])
    Aeq = sparse.coo_matrix(np.hstack([Aeq, np.zeros((Aeq.shape[0],1))]))
    Aiq = sparse.coo_matrix(np.vstack([Aiq,At]))
    Biq = np.concatenate([Biq, np.zeros(N+1)])
    Biq[-1] = -peak_power
    x = optimize.linprog(c, Aiq, Biq, Aeq, Beq, method="highs-ipm")
    # print(x["message"])
    val = x["x"][3:7]
    # val = x["x"][:-1].reshape([-1,7])[:,3:7]
    # val1 = x["x"][:-1].reshape([-1,7])
    # print(x["fun"])
    # return np.array([val[:,0]-val[:,1], val[:,2]-val[:,3]]), val1[:,0]
    return np.array((val[0]-val[1], val[2]-val[3]))


