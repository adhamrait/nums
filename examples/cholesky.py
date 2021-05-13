from nums.core.application_manager import instance
from nums.core.array.blockarray import BlockArray, Block
from nums.core.array.application import ArrayApplication
from nums.core.systems.systems import System
from nums.core.storage.storage import ArrayGrid
from nums.core import settings

import time
import numpy as np
from scipy.linalg import cholesky as scipy_cholesky

settings.system_name = "serial"
app: ArrayApplication = instance()
system: System = app.system

X: BlockArray = BlockArray.from_np(
    np.random.rand(4, 4),
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

def expected_cholesky(A: BlockArray):
    U_scipy = scipy_cholesky(A.get(), lower=False)
    # assert (p == np.eye(p.shape[0])).all()
    U: BlockArray = BlockArray.from_np(
        U_scipy,
        block_shape=(2, 2),
        copy=True,
        system=system,
    )
    return U

def serial_cholesky(A: BlockArray):
    a = A.get()
    assert a.shape[0] == a.shape[1]
    n = a.shape[0]
    u = np.zeros_like(a)

    for k in range(n):
        u[k, k] = np.sqrt(a[k, k])
        u[k, k + 1:] = a[k, k + 1:] / u[k, k]
        for j in range(k + 1, n):
            a[j, j:] = a[j, j:] - u[k, j] * u[k, j:]

    U: BlockArray = BlockArray.from_np(
        u,
        block_shape=(2, 2),
        copy=True,
        system=system,
    )
    return U

def parallel_lu(A: BlockArray):
    grid: ArrayGrid = A.grid.copy()
    L: BlockArray = app.zeros(shape=A.shape, block_shape=A.block_shape)
    U: BlockArray = app.zeros(shape=A.shape, block_shape=A.block_shape)

    for grid_entry in grid.get_entry_iterator():
        print(grid_entry)


    return (L, U)

def raw_cholesky(A):
    if A.shape[0] == 1:
        return np.sqrt(A)
    else:
        n = A.shape[0]
        n2 = n // 2
        A_TL = A[:n2,:n2]
        A_TR = A[:n2,n2:]
        A_BR = A[n2:,n2:]
        R_TL = raw_cholesky(A_TL)
        R_TR = np.linalg.inv(R_TL).T @ A_TR
        R_BR = raw_cholesky(A_BR - R_TR.T @ R_TR)
        result = np.zeros_like(A)
        result[:n2,:n2] = R_TL
        result[:n2,n2:] = R_TR
        result[n2:,n2:] = R_BR
        return result
    

def cholesky_blk(A, b = 2):
    n = A.shape[0]
    if n <= b:
        return scipy_cholesky(A)
    else:
        # initial partition
        A_TL = A[:0,:0]
        A_TR = A[:0,0:]
        A_BR = A[0:,0:]

        while A_TL.shape[0] < n:
            # repartition
            A_00 = A_TL
            A_01 = A_TR[:,:b]
            A_02 = A_TR[:,b:]
            A_11 = A_BR[:b,:b]
            A_12 = A_BR[:b,b:]
            A_22 = A_BR[b:,b:]

            #Variant 1
            # A_01 = np.linalg.inv(A_00).T @ A_00
            # A_11 = A_11 - A_01.T @ A_01
            # A_11 = scipy_cholesky(A_11)

            #Variant 2
            # A_11 = A_11 - A_01.T @ A_01
            # A_11 = scipy_cholesky(A_11)
            # A_12 = A_12 - A_01.T @ A_02
            # A_12 = np.linalg.inv(A_11).T @ A_12

            #Variant 3
            A_11 = scipy_cholesky(A_11)
            A_12 = np.linalg.inv(A_11).T @ A_12
            A_22 = A_22 - A_12.T @ A_12

            # Continue with
            n_00 = A_00.shape[0]
            n_TL = A_00.shape[0] + b
            A_TL = np.zeros((n_TL, n_TL))
            A_TL[:n_00,:n_00] = A_00
            A_TL[:n_00,n_00:] = A_01
            A_TL[n_00:,n_00:] = A_11
            A_TR = np.zeros((n_TL, A_TR.shape[1]-b))
            A_TR[:n_00] = A_02
            A_TR[n_00:] = A_12
            A_BR = A_22
        result = np.zeros_like(A)
        result = A_TL
        return result

def cholesky_blk_nonzero(A, b = 2):
    n = A.shape[0]
    if n <= b:
        return scipy_cholesky(A)
    else:
        # initial partition
        A_TL = A[:b,:b]
        A_TL = scipy_cholesky(A_TL)
        A_TR = A[:b,b:]
        A_TR = np.linalg.inv(A_TL).T @ A_TR # uppertri inv
        A_BR = A[b:,b:]
        A_BR = A_BR - A_TR.T @ A_TR

        while A_TL.shape[0] < n:
            # repartition
            A_00 = A_TL
            A_01 = A_TR[:,:b]
            A_02 = A_TR[:,b:]
            A_11 = A_BR[:b,:b]
            A_12 = A_BR[:b,b:]
            A_22 = A_BR[b:,b:]

            #Variant 3
            A_11 = scipy_cholesky(A_11)
            A_12 = np.linalg.inv(A_11).T @ A_12
            A_22 = A_22 - A_12.T @ A_12

            # Continue with
            n_00 = A_00.shape[0]
            n_TL = A_00.shape[0] + b
            A_TL = np.zeros((n_TL, n_TL))
            A_TL[:n_00,:n_00] = A_00
            A_TL[:n_00,n_00:] = A_01
            A_TL[n_00:,n_00:] = A_11
            A_TR = np.zeros((n_TL, A_TR.shape[1]-b))
            A_TR[:n_00] = A_02
            A_TR[n_00:] = A_12
            A_BR = A_22
        result = np.zeros_like(A)
        result = A_TL
        return result

def inv_uppertri(self, X: BlockArray):
    # Inversion of an Upper Triangular Matrix
    # Use the method described in https://www.cs.utexas.edu/users/flame/pubs/siam_spd.pdf
    assert X.shape[0] == X.shape[1], "This function only accepts square matrices"
    single_block = X.shape[0] == X.block_shape[0] and X.shape[1] == X.block_shape[1]
    nonsquare_block = X.block_shape[0] != X.block_shape[1]

    if single_block or nonsquare_block:
        X = X.reshape(block_shape=(16, 16))

    # Setup metadata
    full_shape = X.shape
    grid_shape = X.grid.grid_shape
    block_shape = X.block_shape

    R = X.copy()
    Zs = self.zeros(full_shape, block_shape, X.dtype)
    
    # Calculate R_11^-1
    r11_oid = R.blocks[(0,0)].oid
    r11_inv_oid = self.system.inv(r11_oid, syskwargs={
                                                "grid_entry": (0, 0),
                                                "grid_shape": grid_shape
                                            })
    R.blocks[(0,0)].oid = r11_inv_oid
    R_tl_shape = block_shape

    # Continue while R_tl.shape != R.shape
    while R_tl_shape[0] != full_shape[0] and R_tl_shape[1] != full_shape[1]:
        # Calculate R11
        R11_block = (int(np.ceil(R_tl_shape[0] // block_shape[0])), int(np.ceil(R_tl_shape[1] // block_shape[1])))
        R11_oid = R.blocks[R11_block].oid
        R11_shape = R.blocks[R11_block].shape

        R11_inv_oid = self.system.inv(R11_oid, syskwargs={
                                                        "grid_entry": R11_block,
                                                        "grid_shape": grid_shape
                                                    })

        # Reset R11 inplace
        R.blocks[R11_block].oid = R11_inv_oid

        # Calculate R01
        R01_oids = []
        R01_shapes = []
        R01_grid_entries = []
        R01_sb_row, R01_sb_col = 0, R11_block[1] # sb -- start_block
        R01_num_blocks = R11_block[0]
        
        # Collect data for R01
        for inc in range(R01_num_blocks):
            R01_oids.append(R.blocks[(R01_sb_row + inc, R01_sb_col)].oid)
            R01_shapes.append(R.blocks[(R01_sb_row + inc, R01_sb_col)].shape)
            R01_grid_entries.append((R01_sb_row + inc, R01_sb_col))
        
        # Perform matrix multiplication: R01_1 = -R00 @ R01
        R01_1_oids = []
        for row_block in range(R01_num_blocks):
            sub_oids = []

            for col_block in range(R01_num_blocks):

                # Get data for R01 and R00
                R01_block_oid = R01_oids[col_block]
                
                R00_oid = R.blocks[(row_block, col_block)].oid
                Z_oid = Zs.blocks[(row_block, col_block)].oid

                R00_bs = R.blocks[(row_block, col_block)].shape

                # Calculate -R00 = 0 - R00
                neg_R00_oid = self.system.bop("subtract", Z_oid, R00_oid, R00_bs, R00_bs, 
                                                False, False, axes=1, syskwargs={
                                                    "grid_entry": (row_block, col_block),
                                                    "grid_shape": grid_shape
                                                })
                # Calculate -R00 @ R01
                sub_oids.append(self.system.bop("tensordot", neg_R00_oid, R01_oids[col_block], 
                                    R00_bs, R01_shapes[col_block], False, False, axes=1, syskwargs={
                                        "grid_entry": R01_grid_entries[col_block],
                                        "grid_shape": grid_shape
                                    }
                                ))
                
            # Finished with one blocked mult
            R01_1_oids.append(self.system.sum_reduce(*sub_oids, syskwargs={
                "grid_entry": R01_grid_entries[row_block],
                "grid_shape": grid_shape
            }))

        # Perform matrix multiplication: R_01_2 = R_01_1 @ R_11_inv
        R01_2_oids = []
        for row_block in range(R01_num_blocks):
            R01_2_oids.append(self.system.bop("tensordot", R01_1_oids[row_block], R11_inv_oid, 
                                R01_shapes[row_block], R11_shape, False, False, axes=1, syskwargs={
                                    "grid_entry": R01_grid_entries[row_block],
                                    "grid_shape": grid_shape
                                }
                            ))

        # Reset R_01
        for i, entry in enumerate(R01_grid_entries):
            R.blocks[entry].oid = R01_2_oids[i]

        # Recompute R_tl.shape 
        r11_r, r11_c = R11_shape
        old_r, old_c = R_tl_shape

        R_tl_shape = (old_r + r11_r, old_c + r11_c)

    # By the time we finish, R = R_inv
    return R


def lu_block_decompose(self, X: BlockArray):
    grid = X.grid.copy()
    # P: BlockArray = BlockArray.from_np(np.zeros(X.shape), grid.block_shape, X.dtype, self.system)
    P: BlockArray = BlockArray(grid, self.system)
    L: BlockArray = BlockArray(grid, self.system)
    U: BlockArray = BlockArray(grid, self.system)
    if len(X.blocks) == 1:
        # Only one block, perform single-block lu decomp
        X_block: Block = X.blocks[0, 0]
        start_small = time.time()
        P.blocks[0, 0].oid, L.blocks[0, 0].oid, U.blocks[0, 0].oid = self.system.lu_inv(X.blocks[0, 0].oid,
            syskwargs={"grid_entry": X_block.grid_entry, "grid_shape": X_block.grid_shape})
        self.block_lu += time.time() - start_small
    else:
        # Must do blocked LU decomp
        size = X.blocks.shape[0]//2
        # sanity check to ensure nice even recursion
        assert size * 2 == X.blocks.shape[0]
        subshape = (X.shape[0]//2, X.shape[1]//2)
        M1 = BlockArray.from_blocks(X.blocks[:size, :size], subshape, self.system)
        M2 = BlockArray.from_blocks(X.blocks[:size, size:], subshape, self.system)
        M3 = BlockArray.from_blocks(X.blocks[size:, :size], subshape, self.system)
        M4 = BlockArray.from_blocks(X.blocks[size:, size:], subshape, self.system)
        
        start_matmul = time.time()
        P1, L1, U1 = self.lu_block_decompose(M1)
        T = U1 @ L1
        Shat = M3 @ T
        Mhat = M4 - Shat @ (P1 @ M2)
        P2, L3, U3 = self.lu_block_decompose(Mhat)
        S = P2 @ Shat
        self.mat_mul += time.time() - start_matmul

        mat_gen = time.time()
        L.blocks[:size, :size] = L1.blocks
        L.blocks[size:, :size] = (-L3 @ S).blocks
        L.blocks[size:, size:] = L3.blocks
        for block_row in L.blocks[:size, size:]:
            for block in block_row:
                syskwargs = {"grid_entry": block.grid_entry, "grid_shape": grid.grid_shape}
                block.oid = self.system.new_block("zeros",
                    block.grid_entry,
                    grid.to_meta(),
                    syskwargs=syskwargs)

        U.blocks[:size, :size] = U1.blocks
        U.blocks[:size, size:] = (-T @ (P1 @ M2) @ U3).blocks
        U.blocks[size:, size:] = U3.blocks
        for block_row in U.blocks[size:, :size]:
            for block in block_row:
                syskwargs = {"grid_entry": block.grid_entry, "grid_shape": grid.grid_shape}
                block.oid = self.system.new_block("zeros",
                    block.grid_entry,
                    grid.to_meta(),
                    syskwargs=syskwargs)

        P.blocks[:size, :size] = P1.blocks
        P.blocks[size:, size:] = P2.blocks
        for block_row in P.blocks[:size, size:]:
            for block in block_row:
                syskwargs = {"grid_entry": block.grid_entry, "grid_shape": grid.grid_shape}
                block.oid = self.system.new_block("zeros",
                    block.grid_entry,
                    grid.to_meta(),
                    syskwargs=syskwargs)
        for block_row in P.blocks[size:, :size]:
            for block in block_row:
                syskwargs = {"grid_entry": block.grid_entry, "grid_shape": grid.grid_shape}
                block.oid = self.system.new_block("zeros",
                    block.grid_entry,
                    grid.to_meta(),
                    syskwargs=syskwargs)
        self.mat_creation += time.time() - mat_gen
    return P, L, U


# print("X: \n", X.get())
# U_ser = serial_cholesky(X)
# U_exp = expected_cholesky(X)

# # check to see if the serial implementation is correct
# if not bool(app.allclose(U_exp, U_ser)):
#     print("U_exp: \n", U_exp.get())
#     print("U_ser: \n", U_ser.get())
#     print("/n")


# U_par = parallel_lu(X)
# # check to see if the parallel implementation is correct
# if not bool(app.allclose(U_exp, U_par)):
#     print("U_exp: \n", U_exp.get())
#     print("/n")
#     print("U_par: \n", U_par.get())
