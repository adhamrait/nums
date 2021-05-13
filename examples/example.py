from nums.core.application_manager import instance
from nums.core.array.blockarray import BlockArray, Block
from nums.core.array.application import ArrayApplication
from nums.core.systems.systems import System
from nums.core.storage.storage import ArrayGrid
from nums.core import settings

import numpy as np

settings.system_name = "serial"
app: ArrayApplication = instance()
system: System = app.system

# r = system.newfun(1, 2, syskwargs={"grid_entry": (0, 0),
#                                    "grid_shape": (1, 1)})
# print(system.get(r))


X: BlockArray = app.arange(shape=(16,), block_shape=(16,)).reshape((4, 4), block_shape=(2, 2))
Y: BlockArray = app.random.random(shape=(4, 4), block_shape=(2, 2))

print(X.get())

grid: ArrayGrid = X.grid.copy()
result: BlockArray = BlockArray(grid, system)


# for grid_entry in grid.get_entry_iterator():
#     x: Block = X.blocks[grid_entry]
#     y: Block = Y.blocks[grid_entry]
#     r: Block = result.blocks[grid_entry]
#     r.oid = system.add(x.oid, y.oid, syskwargs={
#         "grid_entry": grid_entry,
#         "grid_shape": grid.grid_shape
#     })

# print(bool(app.allclose(result, X+Y)))


cluster_shape = np.array(settings.cluster_shape)
for grid_entry in grid.get_entry_iterator():
    x: Block = X.blocks[grid_entry]
    y: Block = Y.blocks[grid_entry]
    r: Block = result.blocks[grid_entry]
    cluster_entry = tuple(np.array(grid_entry) % cluster_shape)
    print(grid_entry, cluster_entry, cluster_shape)
    options = system.get_options(cluster_entry=cluster_entry, cluster_shape=cluster_shape)
    r.oid = system.call_with_options("add",
                                     args=(x.oid, y.oid),
                                     kwargs={}, options=options)

print(bool(app.allclose(result, X+Y)))
