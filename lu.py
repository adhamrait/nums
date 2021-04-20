from nums.core.application_manager import instance
from nums.core.array.blockarray import BlockArray, Block
from nums.core.array.application import ArrayApplication
from nums.core.systems.systems import System
from nums.core.storage.storage import ArrayGrid
from nums.core import settings

import numpy as np
from scipy.linalg import lu as scipy_lu

settings.system_name = "serial"
app: ArrayApplication = instance()
system: System = app.system

X: BlockArray = BlockArray.from_np(
    np.array([[1, 0, 1, 0,], [1, 1, 0, 0,], [1, 1, 1, 1,], [0, 0, 0, 1,]]),
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

def expected_lu(A: BlockArray):
    p, l, u = scipy_lu(A.get())
    # For now, we are not pivoting...
    assert (p == np.eye(p.shape[0])).all()
    L: BlockArray = BlockArray.from_np(
        l,
        block_shape=(2, 2),
        copy=True,
        system=system,
    )
    U: BlockArray = BlockArray.from_np(
        u,
        block_shape=(2, 2),
        copy=True,
        system=system,
    )
    return (L, U)

def serial_lu(A: BlockArray):
    a = A.get()
    assert a.shape[0] == a.shape[1]
    n = a.shape[0]
    l = np.zeros(a.shape)
    u = np.copy(a)
    for k in range(n - 1):
        for i in range(k, n):
            l[i, k] = a[i, k] / a[k, k]
        for j in range(k, n):
            for i in range(k, n):
                a[i, j] = a[i, j] - l[i, k] * a[k, j]
    L: BlockArray = BlockArray.from_np(
        l,
        block_shape=(2, 2),
        copy=True,
        system=system,
    )
    U: BlockArray = BlockArray.from_np(
        u,
        block_shape=(2, 2),
        copy=True,
        system=system,
    )
    return (L, U)
    
def parallel_lu(A: BlockArray):
    grid: ArrayGrid = A.grid.copy()
    L: BlockArray = app.zeros(shape=A.shape, block_shape=A.block_shape)
    U: BlockArray = app.zeros(shape=A.shape, block_shape=A.block_shape)

    for grid_entry in grid.get_entry_iterator():
        print(grid_entry)
        

    return (L, U)


# print("X: \n", X.get())
L_ser, U_ser = serial_lu(X)
L_exp, U_exp = expected_lu(X)

# check to see if the serial implementation is correct
if not (bool(app.allclose(L_exp, L_ser)) and bool(app.allclose(U_exp, U_ser))):
    print("L_exp: \n", L_exp.get())
    print("U_exp: \n", U_exp.get())
    print("/n")
    print("L_ser: \n", L_ser.get())
    print("U_ser: \n", U_ser.get())


L_par, U_par = parallel_lu(X)
# check to see if the parallel implementation is correct
if not (bool(app.allclose(L_exp, L_par)) and bool(app.allclose(U_exp, U_par))):
    print("L_exp: \n", L_exp.get())
    print("U_exp: \n", U_exp.get())
    print("/n")
    print("L_par: \n", L_par.get())
    print("U_par: \n", U_par.get())
