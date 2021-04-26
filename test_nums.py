import argparse

import ray
import nums.numpy as nps
import nums
from nums.core import settings


def main(address, work_dir, use_head, cluster_shape):
    settings.use_head = use_head
    settings.cluster_shape = tuple(map(lambda x: int(x), ",".split(cluster_shape)))
    print("use_head", use_head)
    print("cluster_shape", cluster_shape)
    print("connecting to head node", address)
    ray.init(**{
        "address": address
    })

    print("running nums operation")
    size = 10**4
    # Memory used is 8 * (10**4)**2
    # So this will be 800MB object.
    x1 = nps.random.randn(size, size)
    x2 = nps.random.randn(size, size)
    result = x1 @ x2
    print(result.touch())
    print("writing result")
    write_result = nums.write(work_dir + "/result", result)
    write_result.get()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', default="")
    parser.add_argument('--work-dir', default="/global/homes/e/elibol/dev/test_ray")
    parser.add_argument('--use-head', action="store_true", help="")
    parser.add_argument('--cluster-shape', default="1,1")
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
