# coding=utf-8
# Copyright (C) 2020 NumS Development Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np

from nums.core.storage.storage import ArrayGrid
from nums.core.systems.systems import System
from nums.core.array import utils as array_utils

block_id_counter = -1


class Block(object):
    # pylint: disable=redefined-builtin, global-statement
    # TODO(hme): Create a base class, and move this concrete class into blockarray.py.
    #  Do this when we implement a SparseBlock object.

    def __init__(self, grid_entry, grid_shape, rect, shape, dtype, transposed, system: System,
                 id=None):
        self._system = system
        self.grid_entry: tuple = grid_entry
        self.grid_shape: tuple = grid_shape
        self.rect: list = rect
        self.oid: np.object = None
        self.shape: tuple = shape
        self.dtype = dtype
        self.num_dims = len(self.rect)
        self.transposed = transposed
        self.id = id
        if self.id is None:
            global block_id_counter
            block_id_counter += 1
            self.id = block_id_counter

    def __repr__(self):
        return "Block(" + str(self.oid) + ")"

    def size(self):
        return np.product(self.shape)

    def copy(self, shallow=True):
        assert shallow, "Only shallow copies are currently supported."
        block = Block(self.grid_entry, self.grid_shape, self.rect, self.shape,
                      self.dtype, self.transposed, self._system)
        block.oid = self.oid
        return block

    def true_grid_entry(self):
        if self.transposed:
            return tuple(reversed(self.grid_entry))
        return self.grid_entry

    def true_grid_shape(self):
        if self.transposed:
            return tuple(reversed(self.grid_shape))
        return self.grid_shape

    def transpose(self):
        # This operation does not move or modify the remote object.
        grid_entryT = tuple(reversed(self.grid_entry))
        grid_shapeT = tuple(reversed(self.grid_shape))
        rectT = list(reversed(self.rect))
        blockT = Block(grid_entry=grid_entryT,
                       grid_shape=grid_shapeT,
                       rect=rectT,
                       shape=tuple(reversed(self.shape)),
                       dtype=self.dtype,
                       transposed=not self.transposed,
                       system=self._system)
        blockT.oid = self.oid
        return blockT


    def swapaxes(self, axis1, axis2):
        block = self.copy()
        grid_entry = list(block.grid_entry)
        grid_shape = list(block.grid_shape)
        shape = list(block.shape)
        rect = block.rect

        grid_entry[axis1], grid_entry[axis2] = grid_entry[axis2], grid_entry[axis1]
        grid_shape[axis1], grid_shape[axis2] = grid_shape[axis2], grid_shape[axis1]
        shape[axis1], shape[axis2] = shape[axis2], shape[axis1]
        rect[axis1], rect[axis2] = rect[axis2], rect[axis1]

        block.grid_entry = tuple(grid_entry)
        block.grid_shape = tuple(grid_shape)
        block.shape = tuple(shape)
        block.rect = rect

        block.oid = self._system.swapaxes(block.oid, axis1, axis2,
                                         syskwargs={
                                             "grid_entry": block.grid_entry,
                                             "grid_shape": block.grid_shape
                                         })
        return block

    def ufunc(self, op_name, options=None):
        return self.uop_map(op_name, options=options)

    def uop_map(self, op_name, args=None, kwargs=None, options=None):
        # This retains transpose.
        block = self.copy()
        block.dtype = array_utils.get_uop_output_type(op_name, self.dtype)
        args = () if args is None else args
        kwargs = {} if kwargs is None else kwargs
        if options is None:
            block.oid = self._system.map_uop(op_name,
                                             self.oid,
                                             args,
                                             kwargs,
                                             syskwargs={
                                                 "grid_entry": block.grid_entry,
                                                 "grid_shape": block.grid_shape
                                             })
        else:
            block.oid = self._system.call_with_options("map_uop",
                                                       [op_name, self.oid,
                                                        args,
                                                        kwargs],
                                                       {},
                                                       options)
        return block

    def reduce_axis(self, op_name, axis, keepdims):
        # TODO (hme): Add options version of this too, but need to fix block imps
        #  so we're not branching between options / no options calls in every method.
        # We lose an axis, so make sure to account for dropped axis in result.
        result_grid_entry = []
        result_grid_shape = []
        result_rect = []
        result_shape = []
        for curr_axis in range(len(self.shape)):
            if curr_axis == axis or axis is None:
                if keepdims:
                    result_grid_entry.append(0)
                    result_grid_shape.append(1)
                    result_rect.append((0, 1))
                    result_shape.append(1)
                continue
            result_grid_entry.append(self.grid_entry[curr_axis])
            result_grid_shape.append(self.grid_shape[curr_axis])
            result_rect.append(self.rect[curr_axis])
            result_shape.append(self.shape[curr_axis])

        dtype = array_utils.get_reduce_output_type(op_name, self.dtype)
        block = Block(grid_entry=tuple(result_grid_entry),
                      grid_shape=tuple(result_grid_shape),
                      rect=result_rect,
                      shape=tuple(result_shape),
                      dtype=dtype,
                      # This is false because we invoke the transpose before
                      # applying the reduction operation, so the resulting
                      # remote object will be transposed.
                      transposed=False,
                      system=self._system)
        block.oid = self._system.reduce_axis(op_name=op_name,
                                             arr=self.oid,
                                             axis=axis,
                                             keepdims=keepdims,
                                             transposed=self.transposed,
                                             syskwargs={
                                                 "grid_entry": block.grid_entry,
                                                 "grid_shape": block.grid_shape
                                             })
        return block

    def bop_reduce(self, op_name, other):
        other: Block = other

        dtype = array_utils.get_reduce_output_type(op_name, self.dtype)
        block = Block(grid_entry=self.grid_entry,
                      grid_shape=self.grid_shape,
                      rect=list(self.rect),
                      shape=self.shape,
                      dtype=dtype,
                      # This is false because we invoke the transpose before
                      # applying the reduction operation, so the resulting
                      # remote object will be transposed.
                      transposed=False,
                      system=self._system)
        block.oid = self._system.bop_reduce(op=op_name,
                                            a1=self.oid,
                                            a2=other.oid,
                                            a1_T=self.transposed,
                                            a2_T=other.transposed,
                                            syskwargs={
                                                "grid_entry": block.grid_entry,
                                                "grid_shape": block.grid_shape
                                            })
        return block

    def _block_from_other(self, other):
        # Assume other is numeric.
        # This only occurs during some numpy operations (e.g. np.mean),
        # where a literal is used in the operation.
        assert isinstance(other, (int, float, np.int, np.float))
        block = Block(self.grid_entry, self.grid_shape, [(0, 1)], (1,),
                      self.dtype, False, self._system)
        block.oid = self._system.put(np.array(other, dtype=self.dtype))
        return block

    def bop(self, op, other, args: dict, options=None):
        if not isinstance(other, Block):
            other = self._block_from_other(other)
        if op == "tensordot":
            axes = args["axes"]
            result_grid_entry = tuple(list(self.grid_entry[:-axes]) + list(other.grid_entry[axes:]))
            result_grid_shape = tuple(list(self.grid_shape[:-axes]) + list(other.grid_shape[axes:]))
            result_rect = list(self.rect[:-axes] + other.rect[axes:])
            result_shape = tuple(list(self.shape[:-axes]) + list(other.shape[axes:]))
        else:
            # Broadcasting starts from trailing dimensions.
            # Resulting shape is max of trailing shapes
            result_grid_entry = []
            result_grid_shape = []
            result_rect = []
            result_shape = []
            for i in range(1, max(self.num_dims, other.num_dims) + 1):
                other_i = other.num_dims - i
                self_i = self.num_dims - i
                if other_i < 0:
                    is_self = True
                elif self_i < 0:
                    is_self = False
                else:
                    is_self = other.shape[other_i] < self.shape[self_i]
                if is_self:
                    result_grid_entry.append(self.grid_entry[self_i])
                    result_grid_shape.append(self.grid_shape[self_i])
                    result_rect.append(self.rect[self_i])
                    result_shape.append(self.shape[self_i])
                else:
                    result_grid_entry.append(other.grid_entry[other_i])
                    result_grid_shape.append(other.grid_shape[other_i])
                    result_rect.append(other.rect[other_i])
                    result_shape.append(other.shape[other_i])
            result_grid_entry = tuple(reversed(result_grid_entry))
            result_grid_shape = tuple(reversed(result_grid_shape))
            result_rect = list(reversed(result_rect))
            result_shape = tuple(reversed(result_shape))

        dtype = array_utils.get_bop_output_type(op,
                                                self.dtype,
                                                other.dtype)
        block = Block(grid_entry=result_grid_entry,
                      grid_shape=result_grid_shape,
                      rect=result_rect,
                      shape=result_shape,
                      dtype=dtype,
                      transposed=False,
                      system=self._system)

        # TODO (hme): Get rid of requiring shape as param.
        if options is None:
            block.oid = self._system.bop(op,
                                         self.oid,
                                         other.oid,
                                         self.shape,
                                         other.shape,
                                         self.transposed,
                                         other.transposed,
                                         axes=args.get("axes"),
                                         syskwargs={
                                             "grid_entry": block.grid_entry,
                                             "grid_shape": block.grid_shape
                                         })
        else:
            block.oid = self._system.call_with_options("bop",
                                                       [
                                                           op,
                                                           self.oid,
                                                           other.oid,
                                                           self.shape,
                                                           other.shape,
                                                           self.transposed,
                                                           other.transposed
                                                       ], {
                                                           "axes": args.get("axes")
                                                       },
                                                       options)
        return block

    def tensordot(self, other, axes):
        return self.bop("tensordot", other, args={"axes": axes})

    def __add__(self, other):
        return self.bop("add", other, args={})

    def __sub__(self, other):
        return self.bop("sub", other, args={})

    def __mul__(self, other):
        return self.bop("mul", other, args={})

    def __matmul__(self, other):
        return self.tensordot(other, axes=1)

    def __truediv__(self, other):
        return self.bop("truediv", other, args={})

    def __pow__(self, other):
        return self.bop("pow", other, args={})

    def __ge__(self, other):
        return self.bop("ge", other, args={})

    def __gt__(self, other):
        return self.bop("gt", other, args={})

    def __le__(self, other):
        return self.bop("le", other, args={})

    def __lt__(self, other):
        return self.bop("lt", other, args={})

    def __eq__(self, other):
        return self.bop("eq", other, args={})

    def __ne__(self, other):
        return self.bop("ne", other, args={})

    __iadd__ = __add__
    __isub__ = __sub__
    __imul__ = __mul__
    __imatmul__ = __matmul__
    __itruediv__ = __truediv__
    __ipow__ = __truediv__

    def astype(self, dtype):
        block = self.copy()
        block.dtype = dtype
        block.oid = self._system.astype(self.oid,
                                        dtype.__name__,
                                        syskwargs={
                                            "grid_entry": block.grid_entry,
                                            "grid_shape": block.grid_shape
                                        })
        return block

    def conjugate(self):
        return self.ufunc("conjugate")

    def sqrt(self):
        return self.ufunc("sqrt")

    def syskwargs(self):
        # TODO (hme): This has a lot of potential scheduling bugs:
        #  What if this block is transposed?
        return {
            "grid_entry": self.grid_entry,
            "grid_shape": self.grid_shape
        }

    def get(self):
        return self._system.get(self.oid)


class BlockArrayBase(object):

    def __init__(self, grid: ArrayGrid, system: System, blocks: np.ndarray = None):
        self.grid = grid
        self.system = system
        self.shape = self.grid.shape
        self.block_shape = self.grid.block_shape
        self.size = np.product(self.shape)
        self.ndim = len(self.shape)
        self.dtype = self.grid.dtype
        self.blocks = blocks
        if self.blocks is None:
            # TODO (hme): Subclass np.ndarray for self.blocks instances,
            #  and override key methods to better integrate with NumPy's ufuncs.
            self.blocks = np.empty(shape=self.grid.grid_shape, dtype=Block)
            for grid_entry in self.grid.get_entry_iterator():
                self.blocks[grid_entry] = Block(grid_entry=grid_entry,
                                                grid_shape=self.grid.grid_shape,
                                                rect=self.grid.get_slice_tuples(grid_entry),
                                                shape=self.grid.get_block_shape(grid_entry),
                                                dtype=self.dtype,
                                                transposed=False,
                                                system=self.system)

    def __repr__(self):
        return "BlockArray(" + str(self.blocks) + ")"

    def get(self) -> np.ndarray:
        result: np.ndarray = np.zeros(shape=self.grid.shape, dtype=self.grid.dtype)
        block_shape: np.ndarray = np.array(self.grid.block_shape, dtype=np.int)
        arrays: list = self.system.get([self.blocks[grid_entry].oid
                                        for grid_entry in self.grid.get_entry_iterator()])
        for block_index, grid_entry in enumerate(self.grid.get_entry_iterator()):
            start = block_shape * grid_entry
            entry_shape = np.array(self.grid.get_block_shape(grid_entry), dtype=np.int)
            end = start + entry_shape
            slices = tuple(map(lambda item: slice(*item), zip(*(start, end))))
            block: Block = self.blocks[grid_entry]
            arr: np.ndarray = arrays[block_index]
            if block.transposed:
                arr = arr.T
            result[slices] = arr.reshape(block.shape)
        return result

    def broadcast_to(self, shape):
        b = array_utils.broadcast(self.shape, shape)
        result_block_shape = array_utils.broadcast_block_shape(self.shape, shape, self.block_shape)
        result: BlockArrayBase = BlockArrayBase(ArrayGrid(b.shape,
                                                          result_block_shape,
                                                          self.grid.dtype.__name__), self.system)
        extras = []
        # Below taken directly from _broadcast_to in numpy's stride_tricks.py.
        it = np.nditer(
            (self.blocks,), flags=['multi_index', 'refs_ok', 'zerosize_ok'] + extras,
            op_flags=['readonly'], itershape=result.grid.grid_shape, order='C')
        with it:
            # never really has writebackifcopy semantics
            broadcast = it.itviews[0]
        result.blocks = broadcast
        return result
