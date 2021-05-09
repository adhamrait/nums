import argparse

import ray
import nums.numpy as nps
import nums
import time
from nums.core import settings


def main(address, work_dir, use_head, cluster_shape):
    settings.use_head = use_head
    settings.cluster_shape = tuple(map(lambda x: int(x), cluster_shape.split(",")))
    print("use_head", use_head)
    print("cluster_shape", cluster_shape)
    print("connecting to head node", address)
    ray.init(**{
        "address": address
    })

    print("running nums operation")

    b = 1024
    n = b * 2 ** 4
    A = nps.random.rand(n, n//1024).reshape(block_shape=(b, b))
    t_start = time.time()
    # Put experiment here

    X = (A @ A.T).reshape(block_shape=(b, b))
    _ = nps.linalg.lu_inv(X)

    # Ecperiment done
    t_elapsed = time.time() - t_start
    print(str([n, t_elapsed]) + ",")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', default="")
    parser.add_argument('--work-dir', default="/global/homes/e/elibol/dev/test_ray")
    parser.add_argument('--use-head', action="store_true", help="")
    parser.add_argument('--cluster-shape', default="1,1")
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
