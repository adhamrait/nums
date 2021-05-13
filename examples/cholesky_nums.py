from nums.core.application_manager import instance
from nums.core.array.blockarray import BlockArray, Block
from nums.core.array.application import ArrayApplication
from nums.core.systems.systems import System
from nums.core.storage.storage import ArrayGrid
from nums.core import settings

def cholesky_nums(self,X):
    assert X.shape[0] == X.shape[1], "This function only accepts square matrices"
    assert X.block_shape[0] == X.block_shape[1], "This function only accepts square blocks"
    assert X.shape[0] % X.block_shape[0] == 0, "This function only accepts blocks divisble by size of matrix"
    single_block = X.shape[0] == X.block_shape[0] and X.shape[1] == X.block_shape[1]
    
    # Setup metadata
    full_shape = X.shape
    grid_shape = X.grid.grid_shape
    block_shape = X.block_shape
    
    n,b = full_shape[0], block_shape[0]
    num_blocks = grid_shape[0]
    grid = X.grid.copy()
    if single_block:
        # only one block means we do regular cholesky
        A_TL = self.cholesky(X)
    else:
        # Must do blocked cholesky

        # cholesky on A_TL
        A_TL = BlockArray.from_blocks(X.blocks[:1, :1],(b,b), self.system)
        A_TL = self.cholesky(A_TL)

        # A_TR = inv(A_TL) @ A_TR
        A_TR = BlockArray.from_blocks(X.blocks[:1, 1:], (b,n-b), self.system)
        A_TL_inv = self.inv(A_TL)
        A_TR = A_TL_inv @ A_TR

        # A_BR = A_BR - A_TR.T @ A_TR
        A_BR = BlockArray.from_blocks(X.blocks[1:, 1:],(n-b,n-b), self.system)
        A_BR = A_BR - (A_TR.T @ A_TR)

        while A_TL.shape[0] < n:
            A_TL_size = A_TL.shape[0]
            A_00 = A_TL
            A_01 = BlockArray.from_blocks(A_TR.blocks[:, :1], (A_TL_size,b), self.system)
            A_02 = BlockArray.from_blocks(A_TR.blocks[:, 1:], (A_TL_size,n-A_TL_size-b), self.system)
            A_11 = BlockArray.from_blocks(A_BR.blocks[:1, :1], (b,b), self.system)
            A_12 = BlockArray.from_blocks(A_BR.blocks[:1, 1:], (b,n-A_TL_size-b), self.system)
            A_22 = BlockArray.from_blocks(A_BR.blocks[1:, 1:], (n-A_TL_size-b,n-A_TL_size-b), self.system)

            A_11 = self.cholesky(A_11)

            A_11_inv = self.inv(A_11)
            A_12 = A_11_inv.T, A_12

            A_22 = A_22 - (A_12.T @ A_12)

            # Get new A_TL, A_TR, A_BR
            A_TL = self.zeros((A_TL_size+b,A_TL_size+b),block_shape)
            A_TR = self.zeros((A_TL_size+b,n-A_TL_size-b),block_shape)
            
            for i in range(A_TL_size // b):
                for j in range(A_TL_size // b):
                    A_TL.blocks[i,j].oid = A_00.blocks[i,j].oid

            for i in range(A_TL_size // b):
                A_TL.blocks[i,A_TL_size//b].oid = A_11.blocks[i,0].oid

            A_TL.blocks[A_TL_size//b,A_TL_size//b].oid = A_11.blocks[0,0].oid

            for i in range(A_TL_size // b):
                for j in range((n-A_TL_size-b)//b):
                    A_TR.blocks[i,j].oid = A_02.blocks[i,j].oid

            for j in range((n-A_TL_size-b)//b):
                A_TR.blocks[A_TL_size // b,j].oid = A_02.blocks[0,j].oid

            A_BR = A_22

    return A_TL