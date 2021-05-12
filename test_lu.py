import argparse

from nums.core.application_manager import instance
from nums.core.array.blockarray import BlockArray
from nums.core.array.application import ArrayApplication
from nums.core.systems.systems import System
import time

import numpy as np
from scipy.linalg import lu as scipy_lu


def main(n, b):
  n = int(n)
  b = int(b)
  app: ArrayApplication = instance()
  times = []
  for _ in range(5):
    A = app.random.random((n, n//1024), (b, b))
    t_start = time.time()
    # Put experiment here

    X = (A @ A.T).reshape(block_shape=(b, b))
    _ = app.inv(X)

    # Ecperiment done
    times.append(time.time() - t_start)
  print(str([n, b, times]) + ",")

if __name__ == "__main__":
  for k in range(10):
    for l in range(10):
      b = 1024 * 2 ** l
      main(b * 2 ** k, b)
  print("done!")
