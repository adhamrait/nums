import argparse

import ray
import nums.numpy as nps
import nums
import time
from nums.core import settings


def main(address, work_dir, use_head, cluster_shape):
    settings.use_head = use_head
    settings.cluster_shape = (1,1)
    print("use_head", use_head)
    print("cluster_shape", cluster_shape)
    print("connecting to head node", address)
    ray.init(**{
        "address": address
    })

    print("running nums operation")

    k = 6
    b = 32
    n = b * 2 ** k

    X =  nps.random.rand(n, n).reshape(block_shape=(b, b))
    print(X.block_shape)
    print("starting lu inversion")
    t_st = time.time()
    lu_inverse_par = nps.linalg.lu_inv(X)
    _ = lu_inverse_par.get()
    t_lu = time.time()


    print("starting np inversion")
    expected_inverse = nps.linalg.np_inv(X)
    _ = expected_inverse.get()
    t_ser = time.time()

    print(str([n, b, t_lu-t_st, t_ser-t_lu]) + ",")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', default="")
    parser.add_argument('--work-dir', default="/global/homes/e/elibol/dev/test_ray")
    parser.add_argument('--use-head', action="store_true", help="")
    parser.add_argument('--cluster-shape', default="1,1")
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
