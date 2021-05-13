import time
import ray
from nums.core.application_manager import instance
from nums.core.array.application import ArrayApplication

def main(n, b, c):
  ray.init(**{
      "num_cpus": c
  }) 
  n = int(n)
  b = int(b)
  app: ArrayApplication = instance()
  times = []
  for _ in range(1):
    A = app.random.random((n, n//1024), (b, b))
    # Put experiment here

    X = (A @ A.T).reshape(block_shape=(b, b))
    t_start = time.time()
    _ = app.lu_inv(X)

    # Ecperiment done
    times.append(time.time() - t_start)
  print(str([n, b, c,  times]) + ",")
  ray.shutdown()

if __name__ == "__main__":

  # b = 1024 * 2 ** 4
  # Strong scaling
  n = 1024 * 2 ** 8
  for c in [1, 2, 4, 8, 16, 32, 64]:
    main(n, n // 8, c)
  print("done!")
