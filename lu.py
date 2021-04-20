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

# X: BlockArray = app.random.random(shape=(4,4), block_shape=(2,2))

# Ensure x is stable
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
    l = np.eye(a.shape[0])
    u = np.copy(a)
    for k in range(n):
        for j in range(k+1, n):
            l[j, k] = u[j, k] / u[k, k]
            for i in range(k, n):
                # U (j, k : m) -= L(j, k)U (k, k : m)
                u[j, i] -= l[j, k] * u[k, i]

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

L_ser, U_ser = serial_lu(X)
L_exp, U_exp = expected_lu(X)

# check to see if the serial implementation is correct
if not (bool(app.allclose(L_exp, L_ser)) and bool(app.allclose(U_exp, U_ser))):
    print("L_exp: \n", L_exp.get())
    print("U_exp: \n", U_exp.get())
    print("\n")
    print("L_ser: \n", L_ser.get())
    print("U_ser: \n", U_ser.get())
    print("\n")
    print("X: \n", X.get())


L_par, U_par = app.lu(X)
# check to see if the parallel implementation is correct
if not (bool(app.allclose(L_exp, L_par)) and bool(app.allclose(U_exp, U_par))):
    print("L_exp: \n", L_exp.get())
    print("U_exp: \n", U_exp.get())
    print("\n")
    print("L_par: \n", L_par.get())
    print("U_par: \n", U_par.get())
