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
import numpy as np

class Cholesky_Profiler:
    def __init__(self):
        self.app: ArrayApplication = instance()
        self.system: System = self.app.system

    def generate_PD_matrix(self,n,b):
        A: BlockArray = self.app.random.random(shape=(n, n), block_shape=(b, b))
        A = A.T @ A + self.app.eye(shape=(n, n), block_shape=(b, b))
        print("Generated PD matrix A with shape", A.shape, "and block shape", A.block_shape)
        return A

    def generate_TS_matrix(self,rows,cols,b):
        A: BlockArray = self.app.random.random(shape=(rows, cols), block_shape=(b, b))
        print("Generated TS matrix A with shape", A.shape, "and block shape", A.block_shape)
        assert A.block_shape[0] == A.block_shape[1], "Wrong shapes: {}".format(A.block_shape)
        return A

    def ts_cholesky_inv(self,A: BlockArray, expected_inverse: BlockArray = None):
        print("Starting Cholesky Inverse for Tall-Skinny A")
        start = time.time()
        ATA = A.T @ A
        mult_end = time.time()
        ATA_inv = self.app.inv_cholesky(ATA)
        ATA_fetch = ATA_inv.get()
        inv_end = time.time()
        mult_time, inv_time, fetch_time, total_time = mult_end - start, inv_end-mult_end, inv_end-start

        print("Statistics:\Mult: {}s\nInverse: {}s\nTotal: {}s".format(
            mult_time, inv_time, total_time))
        if expected_inverse is not None:
            print(expected_inverse.shape, expected_inverse.block_shape)
            print(A_inv.shape, A_inv.block_shape)
            print("Is it correct?", bool(self.app.allclose(A_inv, expected_inverse)))
            print("Difference", self.app.sum(self.app.abs(A_inv - expected_inverse)).get())
        del A
        del ATA
        del ATA_inv
        return mult_time, inv_time, total_time

    def pd_cholesky_inv(self,A: BlockArray, expected_inverse: BlockArray = None):
        print("Starting Cholesky Inverse")
        start = time.time()

        A_inv = self.app.inv_cholesky(A)
        A_inv.get()
        inv_end = time.time()
        inv_time = inv_end - start
        print("Inverse Time: {}s".format(
            inv_time))

        if expected_inverse is not None:
            print(expected_inverse.shape, expected_inverse.block_shape)
            print(A_inv.shape, A_inv.block_shape)
            print("Is it correct?", bool(self.app.allclose(A_inv, expected_inverse)))
            print("Difference", self.app.sum(self.app.abs(A_inv- expected_inverse)).get())
        del A
        del A_inv
        return inv_time

    def profile_ts(self):
        # seeded_timings = np.array([]).reshape(0,15,6)
        # for seed_num in range(1):
        timings = np.array([]).reshape(0,6)
        for i in range(12,13):
            cols = 2**i
            rows = 1024*cols
            for b_k in range(max(7,i-3),i+1):
                b = 2**b_k
                A = self.generate_TS_matrix(rows,cols,b)
                mult_time, inv_time, total_time = self.ts_cholesky_inv(A)

                times = np.array([[rows,cols,b,mult_time,inv_time,total_time]])
                timings = np.concatenate((timings,times), axis=0)
            # seeded_timings = np.concatenate((seeded_timings,timings),axis=0)
        np.save("/home/ubuntu/nums/outputs/cholesky-ts-4096-64", timings)
        
    def profile_pd(self):
        timings = np.array([]).reshape(0,3)
        for i in range(9,15):
            cols = 2**i
            rows = cols
            for b_k in range(max(7,i-5),i-1):
                b = 2**b_k
                A = self.generate_PD_matrix(cols,b)
                inv_time = self.pd_cholesky_inv(A)

                times = np.array([[rows,b,inv_time]])
                timings = np.concatenate((timings,times), axis=0)
            # seeded_timings = np.concatenate((seeded_timings,timings),axis=0)
        print("Finished Profiling PD")
        np.save("/home/ubuntu/nums/outputs/cholesky-pd-64", timings)
        
    def main(self, address, work_dir, use_head, cluster_shape, features, block_size, cpus):
        # These lines are for multinode
        # settings.use_head = use_head
        # settings.cluster_shape = tuple(map(lambda x: int(x), cluster_shape.split(",")))
        # print("use_head", use_head)
        # print("cluster_shape", cluster_shape)
        # print("connecting to head node", address)
        # ray.init(**{
        #     "address": "172.31.40.240:6397",
        # }) 

        output_file = "outputs/cholesky/choleskyinv-1-{}-{}-{}.txt".format(cpus, features, block_size)

        print("running nums operation")
        # output_file.write("running nums operation\n")

        cols = features
        rows = features * 1024
        b = block_size

        print("Beginning experiment with {} Cluster Shape, {} Features, and {} Block Size".format(cluster_shape, cols, b))
        # output_file.write("Beginning experiment with {} Cluster Shape, {} Features, and {} Block Size\n".format(cluster_shape, cols, b))

        # output_file.write("Generated PD matrix A with shape {} and block shape {}\n".format(A.shape, A.block_shape))

        """
            Main Experiment:

            Perform Direct TSQR Factorization
            Perform Upper Triangular Matrix 
            Perform Matrix Multiplication
        """ 
        expected_inverse = app.inv(A)
        cholesky_inv(app, A, expected_inverse, output_file)
        output_file.close()
        print("finished")
        ray.shutdown()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--address', default="")
    # parser.add_argument('--work-dir', default="~/nums/outputs")
    # parser.add_argument('--use-head', action="store_true", help="")
    # parser.add_argument('--cluster-shape', default="1,1")
    # parser.add_argument('--features', type=int, default=128)
    # parser.add_argument('--block-size', type=int, default=64)
    # parser.add_argument('--cpus', type=int, default=64)
    # args = parser.parse_args()
    # kwargs = vars(args)
    # main(**kwargs)
    profiler = Cholesky_Profiler()
    # profiler.profile_ts()
    profiler.profile_pd()