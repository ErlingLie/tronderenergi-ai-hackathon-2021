import numpy as np
from scipy import optimize
from scipy import linalg
from cvxopt import matrix, solvers

def generateIQMatrics():
    A = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    B = np.array([[0.85, 0, 0, 0], [ 0, 1, 0, 0], [0, 0, 0.325, 0], [0, 0, 0, 1]])

    C = -np.eye(4)

    b = np.array([500, 0, 1670, 0])
    c = np.array([0, 0, 0, 0, 400, 400, 55, 100])

    C = np.vstack([C, -C])
    C_big = np.kron(np.eye(7), C)
    c_big = np.kron(np.ones(7), c)
    structure = np.zeros((28,7))
    structure[0:3, 0:3] = np.eye(3)
    structure[3:5, 3] = 1
    structure[5:9, 4] = 1
    structure[9:17, 5] = 1
    structure[17:, 6 ] = 1

    A_big = np.kron(np.eye(28), A)
    B_big = np.kron(structure, B)
    b_big = np.kron(np.ones(structure.shape[0]), b)

    Cx = np.zeros((C_big.shape[0], 3*28))
    A = np.block([[Cx, C_big], [A_big, B_big]])

    B = np.hstack([c_big, b_big])

    return A, B
def genFullIQMatrix(N):
    return generateIQMatrics()


def generateEQInitalMatrix(x0):
    A1 = np.array([[1, 0, 0, 0.85, -1, 0, 0], [0, 1, 0, 0, 0, 0.325, -1]])
    A1[:, 3:] *= -1
    b0 = np.array([[1, 0, 0], [0,1,0]])@x0
    return A1[:,:3], A1[:,3:], b0

def generateEQMatrices(PV, W, C):
    A1 = np.array([[1, 0, 0, 0.85, -1, 0, 0], [0, 1, 0, 0, 0, 0.325, -1]])
    B1 = np.zeros((2,3))
    B1[:2,:2] = -np.eye(2)
    Aeqx = np.hstack([A1[:,:3], B1])
    b1 = np.array([0,0])

    Aeq2 = np.array([[0, 0, -1, 1, -1, 1, -1]])
    b2 = np.array([PV + W - C])

    Aeqx2 = np.hstack([Aeq2[:, :3], np.zeros((1,3))])
    Ax = np.vstack([Aeqx, Aeqx2])
    b = np.hstack([b1, b2])
    Au = np.vstack([A1[:,3:], Aeq2[:,3:]])

    return Ax, Au, b

def genQ(N, spot):
    Q = []
    for i in range(28):
        q = np.zeros((3,3))
        q[2,2] = 0.05 + spot[i]
        Q.append(q)
    for i in range(7):
        Q.append(np.zeros((4,4)))

    return linalg.block_diag(*Q)

def genFullMatrix(N, x0, PV, W, C):
    #matrix = np.zeros(((N-1)*3+2, 7*N))
    
    Ax, Au, b0 = generateEQInitalMatrix(x0)
    initialMatrix1 = np.zeros((Ax.shape[0],28*3))
    initialMatrix1[:,:3] = Ax
    initialMatrix2 = np.zeros((Ax.shape[0], 4*7))
    initialMatrix2[:, :4] = Au
   # print(A.shape, b.shape, A1.shape, b0.shape)
    Ax, Au, b = generateEQMatrices(PV[0], W[0], C[0])
    A_big = np.zeros((28*3,28*3))
    c = 0
    for i in range(27):
        A_big[c:3+c,c:6+c] = Ax
        c += 3
    #A_big = np.kron(np.eye(28), Ax)
    structure = np.zeros((28, 7))
    structure[0, 1] = 1
    structure[1, 2] = 1
    structure[2, 2] = 1
    structure[3:5, 3] = 1
    structure[5:9, 4] = 1
    structure[9:17, 5] = 1
    structure[17:, 6 ] = 1
    B_big = np.kron(structure, Au)

    #print(initialMatrix1.shape, initialMatrix2.shape, A_big.shape, B_big.shape)
    A = np.block([[initialMatrix1, initialMatrix2], [A_big, B_big]])
    rhs = [b0]
    for pv, w, cons in zip(PV, W, C):
        rhs.append(np.array([0, 0, pv + w - c]))
    b = np.hstack(rhs) 
    return A, b



def MPC_step(N, x0, PV, W, C, spot):
    solvers.options['show_progress'] = False

    Aeq, Beq = genFullMatrix(N, x0, PV, W, C)


    Aiq, Biq = genFullIQMatrix(N)

    Q = genQ(N, spot)
    P  = matrix(Q)
    G = matrix(Aiq)
    h = matrix(Biq)
    A = matrix(Aeq)
    b = matrix(Beq.reshape([-1]).astype(float))


    x = solvers.qp(P, matrix(np.zeros(Q.shape[0]).reshape([-1,1])), G, h, A, b, kktsolver="chol")
    val = x["x"][3*28:3*28+4]
    return np.array((val[0]-val[1], val[2]-val[3]))


