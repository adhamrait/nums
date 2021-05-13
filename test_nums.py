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

def directtsqr_invuppertri(app: ArrayApplication, A: BlockArray, expected_inverse: BlockArray, file):
    print("Starting Direct TSQR -> Upper Triangular Inverse -> Matrix mult [2]")
    start = time.time()

    Q, R = app.qr(A)
    # _ = R.get()
    qr_end = time.time()
    # print("S: {}, B: {}".format(R.shape, R.block_shape))

    # start = time.time()
    R_inv = app.inv_uppertri(R)
    # _ = R_inv.get()
    inv_end = time.time()

    inverse = R_inv @ R_inv.T
    # _ = inverse.get()
    mult_end = time.time()

    _ = inverse.get()
    end = time.time()
    # statistics = "Statistics:\nTSR: {}s\nQ: {}s\nInv: {}s\nMult: {}s".format(tsr_time, q_time, inv_end-start, mult_end-inv_end)
    # file.write(statistics + "\n")
    # print(statistics)
    statistics = "Statistics:\nTSQR: {}s\nUpperTri Inverse: {}s\nMultiplication: {}s\nFetch: {}s\nTotal: {}s".format(
        qr_end-start, inv_end-qr_end, mult_end-inv_end, end-mult_end, end-start
    )
    file.write(statistics + "\n")
    print(statistics)
    if expected_inverse is not None:
        print("Is it correct?", bool(app.allclose(inverse, expected_inverse)))

def baseline_1(app: ArrayApplication, A: BlockArray, expected_inverse: BlockArray, file):
    print("Starting baseline QR factorization -> R naive inverse -> matrix mult [1]")
    start = time.time()

    # print("here")
    Q, R = app.qr(A)
    # _ = R.get()
    qr_end = time.time()
    # print("R block size:")

    R_inv = app.inv(R)
    # _ = R_inv.get()
    inv_end = time.time()
    # print("R_inv block size:")

    inverse = R_inv @ R_inv.T
    mult_end = time.time()
    # print("end")

    _ = inverse.get()
    end = time.time()

    statistics = "Statistics:\nQR: {}s\nInverse: {}s\nMultiplication: {}s\nFetch: {}s\nTotal: {}s".format(
        qr_end-start, inv_end-qr_end, mult_end-inv_end, end-mult_end, end-start
    )
    file.write(statistics + "\n")
    print(statistics)
    if expected_inverse is not None:
        print("Is it correct?", bool(app.allclose(inverse, expected_inverse)))

def baseline_0(app, A):
    print("Starting baseline matrix mult -> inverse [0]")
    start = time.time()
    
    ATA = A.T @ A
    mult_end = time.time()
    expected_inverse = app.inv(ATA)
    inv_end = time.time()
    _ = expected_inverse.get()
    end = time.time()

    print("Statistics:\nMultiplication: {}s\nInverse: {}s\nFetch: {}s\nTotal: {}s".format(
        mult_end-start, inv_end-mult_end, end-inv_end, end-start
    ))

    return expected_inverse

def main(address, work_dir, use_head, cluster_shape, features, block_size, cpus):
    settings.use_head = use_head
    settings.cluster_shape = tuple(map(lambda x: int(x), cluster_shape.split(",")))
    # print("use_head", use_head)
    # print("cluster_shape", cluster_shape)
    print("connecting to head node", address)
    ray.init(**{
        "address": "172.31.40.240:6397",
    })
    output_file = open("outputs/tsqr/teeindtsqr-1-{}-{}-{}.txt".format(cpus, features, block_size), "w")

    print("running nums operation")
    output_file.write("running nums operation\n")

    cols = features
    rows = features * 1024
    b = block_size

    print("Beginning experiment with {} Cluster Shape, {} Features, and {} Block Size".format(cluster_shape, cols, b))
    output_file.write("Beginning experiment with {} Cluster Shape, {} Features, and {} Block Size\n".format(cluster_shape, cols, b))

    app: ArrayApplication = instance()
    system: System = app.system

    A: BlockArray = app.random.random(shape=(rows, cols), block_shape=(b, b))
    print("Generated TS matrix A with shape", A.shape, "and block shape", A.block_shape)
    output_file.write("Generated TS matrix A with shape {} and block shape {}\n".format(A.shape, A.block_shape))
    
    """
        First Baseline Experiment:

        Perform Matrix Multiplication: A.T @ A
        Perform Inversion: (ATA)^-1
    """
    # expected_inverse = baseline_0(app, A)

    """
        Second Baseline Experiment:
        
        Perform QR Factorization
        Perform Naive Inversion of R 
        Perform Matrix Multiplication: R_inv @ R_inv.T
    """
    # baseline_1(app, A, None, output_file)

    """
        Main Experiment:

        Perform Direct TSQR Factorization
        Perform Upper Triangular Matrix 
        Perform Matrix Multiplication
    """ 
    directtsqr_invuppertri(app, A, None, output_file)
    output_file.close()
    print("finished")
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', default="")
    parser.add_argument('--work-dir', default="/global/homes/j/jiwania/nums/outputs")
    parser.add_argument('--use-head', action="store_true", help="")
    parser.add_argument('--cluster-shape', default="1,1")
    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--cpus', type=int, default=64)
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
