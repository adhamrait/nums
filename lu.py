from nums.core.application_manager import instance
from nums.core.array.blockarray import BlockArray
from nums.core.array.application import ArrayApplication
from nums.core.systems.systems import System
import time

import numpy as np
from scipy.linalg import lu as scipy_lu

app: ArrayApplication = instance()
system: System = app.system

np.random.seed(1)
k = 4
b = 256
n = b * 2 ** k

block_shape = (b, b)
X: BlockArray = app.random.random((n, n), block_shape)

def lu_block_decomp(M):
    if M.shape == block_shape:
        p, l, u = scipy_lu(M)
        return(p.T, np.linalg.inv(l), np.linalg.inv(u))
    else:
        size = M.shape[0]//2

        M1 = M[:size, :size]
        M2 = M[:size, size:]
        M3 = M[size:, :size]
        M4 = M[size:, size:]

        P1, L1, U1 = lu_block_decomp(M1)
        U2 = L1.dot(P1.dot(M2))
        L2hat = M3.dot(U1)
        Mhat = M4 - L2hat.dot(U2)
        P2, L3, U3 = lu_block_decomp(Mhat)
        L2 = P2.dot(L2hat)

        L = np.zeros(M.shape)
        U = np.zeros(M.shape)
        P = np.zeros(M.shape)

        L[:size, :size] = L1
        L[size:, :size] = -L3.dot(L2.dot(L1))
        L[size:, size:] = L3

        U[:size, :size] = U1
        U[:size, size:] = -U1.dot(U2.dot(U3))
        U[size:, size:] = U3

        P[:size, :size] = P1
        P[size:, size:] = P2

    return P, L, U

def lu_inv_sequential(X: BlockArray):
    p, l, u = lu_block_decomp(X.get())
    return BlockArray.from_np(
        u.dot(l.dot(p)),
        block_shape=block_shape,
        copy=True,
        system=system,
    )

print("starting lu inversion")
t_st = time.time()
lu_inverse_par = app.lu_inv(X)
_ = lu_inverse_par.get()
t_lu = time.time()


print("starting np inversion")
expected_inverse = app.inv(X)
_ = expected_inverse.get()
t_ser = time.time()

# check to see if the parallel implementation is correct
# if not bool(app.allclose(lu_inverse_par, expected_inverse)):
#     print("INCORRECT RESULTS")
#     print("parallel: \n", lu_inverse_par.get())
#     print("sequential: \n", expected_inverse.get())
#     print("\n")
#     print("X: \n", X.get())
print(str([n, b, t_lu-t_st, t_ser-t_lu]) + ",")
