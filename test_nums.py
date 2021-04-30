import argparse
import time
import ray
import nums.numpy as nps
import nums
from nums.core import settings
from nums.core.application_manager import instance
from nums.core.array.blockarray import BlockArray

from nums.core.array.application import ArrayApplication
from nums.core.systems.systems import System


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

    b = 16
    m = 10000
    n = 1000

    app: ArrayApplication = instance()
    system: System = app.system

    A: BlockArray = app.random.random(shape=(m,n), block_shape=(b,b))
    print("starting lu inversion")

    print("Generated TS matrix A with shape", A.shape, "and block shape", A.block_shape)
    
    print("Starting baseline matrix mult and inverse")
    bmmi_s = time.time()
    
    ATA = A.T @ A
    expected_inverse = app.inv(ATA)
    _ = expected_inverse.get()

    bmmi_e = time.time()
    print("Experiment took", bmmi_e - bmmi_s, "seconds\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', default="")
    parser.add_argument('--work-dir', default="/global/homes/j/jiwania/nums/outputs")
    parser.add_argument('--use-head', action="store_true", help="")
    parser.add_argument('--cluster-shape', default="1,1")
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
