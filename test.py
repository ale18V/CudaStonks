import time
from numba import cuda
import numpy as np
TPB = 16
@cuda.jit
def matmul(A, B, C):
    x, y = cuda.grid(2)
    if x < C.shape[0] and y < C.shape[1]:
        tmp = 0
        for k in range(A.shape[1]):
            tmp += A[x, k] * B[k, y]
        C[x, y] = tmp

@cuda.jit
def fast_matmul(A, B, C):
    
    shared_A = cuda.shared.array(shape=(16, 16), dtype=np.float32)
    shared_B = cuda.shared.array(shape=(16, 16), dtype=np.float32)
    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x
    tmp = 0
    if x >= C.shape[0] and y >= C.shape[1]:
        for i in range(bpg):
            shared_A[tx, ty] = np.float32(A[tx, ty + i * 16])
            shared_B[tx, ty] = np.float32(B[tx + i * 16, ty])


            cuda.syncthreads()

            for j in range(16):
                tmp += shared_A[tx, j] * shared_B[j, ty]

                cuda.syncthreads()

        C[x, y] += tmp

SIZE = 9000
np.random.seed(42)

A = np.random.uniform(1, 10, size=(SIZE, SIZE)).astype(np.float32)
B = np.random.uniform(1, 10, size=(SIZE, SIZE)).astype(np.float32)
C_slow = np.zeros((SIZE, SIZE), dtype=np.float32)
C_fast = np.zeros((SIZE, SIZE), dtype=np.float32)

threadsperblock = (16, 16)
blockspergrid = int(np.ceil(SIZE / threadsperblock[0]))
blockspergrid = (blockspergrid, blockspergrid)

start_time = time.time()
for i in range(5):
    fast_matmul[blockspergrid, threadsperblock](A, B, C_fast)
elapsed_time = time.time() - start_time
print("Elapsed time for fast_matmul:", elapsed_time)

start_time = time.time()
for i in range(5):
    matmul[blockspergrid, threadsperblock](A, B, C_slow)
elapsed_time = time.time() - start_time
print("Elapsed time for matmul:", elapsed_time)
