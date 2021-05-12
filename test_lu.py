import argparse

from nums.core.application_manager import instance
from nums.core.array.blockarray import BlockArray
from nums.core.array.application import ArrayApplication
from nums.core.systems.systems import System
import time


def main(n, b):
  n = int(n)
  b = int(b)
  app: ArrayApplication = instance()
  times = []
  # for _ in range(5):
  A = app.random.random((n, n//1024), (b, b))
  t_start = time.time()
  # Put experiment here

  X = (A @ A.T).reshape(block_shape=(b, b))
  _ = app.lu_inv(X)

  # Ecperiment done
  times.append(time.time() - t_start)
  print(str([n, b, times]) + ",")

if __name__ == "__main__":
  main(4096*4 * 2, 4096 * 4)
  print("done!")
