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

np.random.seed(69)
x = np.random.rand(64, 64)
X: BlockArray = BlockArray.from_np(
    x,
    block_shape=(2, 2),
    copy=True,
    system=system,
)

expected_inverse: BlockArray = BlockArray.from_np(
    np.linalg.inv(X.get()),
    block_shape=(2, 2),
    copy=True,
    system=system,
)

def lu_inv(A: BlockArray):
    p, l, u = scipy_lu(A.get())
    # For now, we are not pivoting...
    l_inv = np.linalg.inv(l)
    u_inv = np.linalg.inv(u)
    inv: BlockArray = BlockArray.from_np(
        u_inv.dot(l_inv.dot(p.T)),
        block_shape=(2, 2),
        copy=True,
        system=system,
    )
    return inv

lu_inverse = lu_inv(X)

# check to see if the parallel implementation is correct
if not bool(app.allclose(lu_inverse, expected_inverse)):
    print("lu_inverse: \n", lu_inverse.get())
    print("expected_inverse: \n", expected_inverse.get())
    print("\n")
    print("X: \n", X.get())

block_shape = X.block_shape

lu_inverse_impl = app.lu_inv(X)

# check to see if the parallel implementation is correct
if not bool(app.allclose(lu_inverse_impl, expected_inverse)):
    print("lu_inverse_impl: \n", lu_inverse_impl.get())
    print("expected_inverse: \n", expected_inverse.get())
    print("\n")
    print("X: \n", X.get())
