from nums.core.application_manager import instance
from nums.core.array.blockarray import BlockArray

from nums.core.array.application import ArrayApplication
from nums.core.systems.systems import System

import numpy as np

app: ArrayApplication = instance()
system: System = app.system

# X: BlockArray = app.random.random(shape=(4,4), block_shape=(2,2))
# for grid_entry in X.grid.get_entry_iterator():
#     print(grid_entry, X.blocks[grid_entry].shape)

# Triangular matrix
a = np.ones((4,4))
r = np.triu(a)

R: BlockArray = app.array(r, block_shape=(2,2))

# Get Inverse
R11_inv = app.temp_ts_inv(R)
print(R11_inv.get())


# Y: BlockArray = app.random.random(shape=(2,2), block_shape=(1,1))

# print(X.get())
# print(Y.get())

# print((X+Y).get())

# first_entry = list(X.grid.get_entry_iterator())[0]
# x_block = X.blocks[first_entry]
# y_block = Y.blocks[first_entry]

# print(x_block.get())
# print(y_block.get())

# result = app.zeros(shape=(2,2), block_shape=(1,1))

# system.call_with_options()

# result.blocks[first_entry].oid = system.bop("add", x_block.oid, y_block.oid, x_block.shape, y_block.shape, False, False, axes=1, syskwargs={"grid_entry": (0,0), "grid_shape": (1,1)})

# print(result.blocks[first_entry].get())

# replacement = app.array(np.ones((1,1)), block_shape=(1,1))
# result.blocks[first_entry].oid = replacement.blocks[(0,0)].oid

# print(result.blocks[first_entry].get())

# result.blocks[first_entry] = np.ones((1,1))
# print(result.blocks[first_entry].get())
# print(X.shape)
# for grid_entry in X.grid.get_entry_iterator():
#     print(X.blocks[grid_entry].get())
#     break