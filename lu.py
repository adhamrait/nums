from nums.core.application_manager import instance
from nums.core.array.blockarray import BlockArray, Block
from nums.core.array.application import ArrayApplication
from nums.core.systems.systems import System
from nums.core import settings

import numpy as np
from scipy.linalg import lu as scipy_lu

settings.system_name = "serial"
app: ArrayApplication = instance()
system: System = app.system

# X: BlockArray = app.random.random(shape=(4,4), block_shape=(2,2))

# Ensure x is stable
np.random.seed(69)
x = np.random.rand(4, 4)
x = scipy_lu(x)[0].T @ x
X: BlockArray = BlockArray.from_np(
    x,
    block_shape=(2, 2),
    copy=True,
    system=system,
)

# expected_inverse: BlockArray = BlockArray.from_np(
#     np.linalg.inv(X.get()),
#     block_shape=(2, 2),
#     copy=T    rue,
#     system=system,
# )

def expected_lu(A: BlockArray):
    p, l, u = scipy_lu(A.get())
    # For now, we are not pivoting...
    assert (p == np.eye(p.shape[0])).all()
    LU: BlockArray = BlockArray.from_np(
        u + l - np.eye(l.shape[0], l.shape[1]),
        block_shape=(2, 2),
        copy=True,
        system=system,
    )
    return LU


# for k in range(n):
#     for j in range(k+1, n):
#         factor = lu[j, k] / lu[k, k]
#         lu[j, k] = factor
#         for i in range(k+1, n):
#             lu[j, i] -= factor * lu[k, i]

def serial_lu(A: BlockArray):
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    lu = np.copy(A.get())

    for k in range(n):
        # row = lu[k, k+1:]
        factors = lu / lu[k, k]
        lu[k+1:, k] = factors[k+1:, k]
        lu[k+1:, k+1:] -= np.outer(factors[:, k], lu[k, :])[k+1:, k+1:]

    LU: BlockArray = BlockArray.from_np(
        lu,
        block_shape=X.block_shape,
        copy=True,
        system=system,
    )
    return LU

LU_ser = serial_lu(X)
LU_exp = expected_lu(X)

# check to see if the serial implementation is correct
if not bool(app.allclose(LU_ser, LU_exp)):
    print("LU_exp: \n", LU_exp.get())
    print("LU_ser: \n", LU_ser.get())
    print("\n")
    print("X: \n", X.get())


# LU_par= app.lu(X)
# # check to see if the parallel implementation is correct
# if not bool(app.allclose(LU_par, LU_exp)):
#     print("LU_exp: \n", LU_exp.get())
#     print("LU_par: \n", LU_par.get())
#     print("\n")
#     print("X: \n", X.get())