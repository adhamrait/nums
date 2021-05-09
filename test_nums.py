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

def directtsqr_invuppertri(app: ArrayApplication, A: BlockArray, expected_inverse: BlockArray):
    print("Starting Direct TSQR -> Upper Triangular Inverse -> Matrix mult [2]")
    start = time.time()

    Q, R = app.direct_tsqr(A)

    R_inv = app.inv_uppertri(R)

    inverse = R_inv @ R_inv.T

    end = time.time()
    print("Experiment took", end - start, "seconds")
    print("Is it correct?", bool(app.allclose(inverse, expected_inverse)))

def baseline_1(app: ArrayApplication, A: BlockArray, expected_inverse: BlockArray):
    print("Starting baseline QR factorization -> R naive inverse -> matrix mult [1]")
    start = time.time()

    Q, R = app.qr(A)

    R_inv = app.inv(R)

    inverse = R_inv @ R_inv.T

    end = time.time()
    print("Experiment took", end - start, "seconds")
    print("Is it correct?", bool(app.allclose(inverse, expected_inverse)))

def baseline_0(app, A):
    print("Starting baseline matrix mult -> inverse [0]")
    start = time.time()
    
    ATA = A.T @ A
    expected_inverse = app.inv(ATA)
    _ = expected_inverse.get()

    end = time.time()
    print("Experiment took", end - start, "seconds\n")

    return expected_inverse

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

    b = 10
    m = 100
    n = 10

    app: ArrayApplication = instance()
    system: System = app.system

    A: BlockArray = app.random.random(shape=(m,n), block_shape=(b,b))
    print("Generated TS matrix A with shape", A.shape, "and block shape", A.block_shape)
    
    """
        First Baseline Experiment:

        Perform Matrix Multiplication: A.T @ A
        Perform Inversion: (ATA)^-1
    """
    expected_inverse = baseline_0(app, A)

    """
        Second Baseline Experiment:
        
        Perform QR Factorization
        Perform Naive Inversion of R 
        Perform Matrix Multiplication: R_inv @ R_inv.T
    """
    baseline_1(app, A, expected_inverse)

    """
        Main Experiment:

        Perform Direct TSQR Factorization
        Perform Upper Triangular Matrix 
        Perform Matrix Multiplication
    """ 
    # directtsqr_invuppertri(app, A, expected_inverse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', default="")
    parser.add_argument('--work-dir', default="/global/homes/j/jiwania/nums/outputs")
    parser.add_argument('--use-head', action="store_true", help="")
    parser.add_argument('--cluster-shape', default="1,1")
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
