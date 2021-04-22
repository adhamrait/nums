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
x = np.random.rand(8, 8)
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

def update_lu(lu_block: np.array, jmin, imin, factor, k):
    for j in range(jmin, lu_block.shape[0]):
        factor = lu_block[j, k] / factor
        lu_block[j, k] = factor
        for i in range(imin, lu_block.shape[1]):
            lu_block[j, i] -= factor * lu_block[k, i]



def serial_lu(A: BlockArray):
    assert A.shape[0] == A.shape[1]
    n = A.shape[0]
    lu = np.copy(A.get())
    grid = A.grid.copy()

    for k in range(n):
        block_col = k//grid.block_shape[0]
        block_row = k//grid.block_shape[1]
        k_col = k%grid.block_shape[0]
        k_row = k%grid.block_shape[1]
        for j_block in range(block_col, grid.block_shape[0]):
            break





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