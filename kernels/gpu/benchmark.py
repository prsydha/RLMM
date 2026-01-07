"""
GPU Baselines benchmark harness using CuPy
Implements:
 - naive matmul (CuPy RawKernel)
 - tiled matmul (shared memory, CuPy RawKernel)
 - cuBLAS matmul (via CuPy)
 - naive convolution via im2col + matmul
 - cuDNN convolution (via CuPy)

Usage:
  python kernels/gpu/benchmark.py

Requirements:
  - Python 3.8+
  - cupy (install matching your CUDA version), e.g. `pip install cupy-cuda12x`
"""

import time
import math
import numpy as np
import cupy as cp
from cupy import RawKernel

# Naive matmul kernel (each thread computes one output element)
naive_mm_kernel = RawKernel(r"""
extern "C" __global__
void naive_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
""", "naive_gemm")

# Tiled (shared memory) matmul kernel: tile sizes 16x16
tiled_mm_kernel = RawKernel(r"""
extern "C" __global__
void tiled_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    const int TS = 16;
    __shared__ float sA[TS][TS];
    __shared__ float sB[TS][TS];

    int row = blockIdx.y * TS + threadIdx.y;
    int col = blockIdx.x * TS + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TS - 1) / TS; ++t) {
        int aRow = row;
        int aCol = t * TS + threadIdx.x;
        int bRow = t * TS + threadIdx.y;
        int bCol = col;

        sA[threadIdx.y][threadIdx.x] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TS; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
""", "tiled_gemm")

# Helper: measure kernel execution with CUDA events
def time_kernel(fn, *args, sync=True):
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    cp.cuda.runtime.deviceSynchronize()
    start.record()
    fn(*args)
    if sync:
        end.record()
        end.synchronize()
        ms = cp.cuda.get_elapsed_time(start, end)
        return ms * 1e-3
    else:
        return None

# Run a raw kernel with grid/block configuration
def run_naive_mm(A, B, C):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    block = (16, 16, 1)
    grid = ((N + block[0]-1)//block[0], (M + block[1]-1)//block[1], 1)
    naive_mm_kernel(grid, block, (A, B, C, M, N, K))

def run_tiled_mm(A, B, C):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    TS = 16
    block = (TS, TS, 1)
    grid = ((N + TS-1)//TS, (M + TS-1)//TS, 1)
    tiled_mm_kernel(grid, block, (A, B, C, M, N, K))

# cuBLAS using cupy.dot (leverages cublas internally)
def run_cublas_mm(A, B, C):
    # using cupy dot to compute into C
    cp.dot(A, B, out=C)

# Naive convolution via im2col + matmul
# For simplicity implement a single-batch, single-input, single-output channel conv (NCHW simplified)
def im2col_gpu(x, kh, kw, stride=1, pad=0):
    # x: (C, H, W)
    C, H, W = x.shape
    out_h = (H + 2*pad - kh)//stride + 1
    out_w = (W + 2*pad - kw)//stride + 1
    cols = cp.zeros((C * kh * kw, out_h * out_w), dtype=cp.float32)
    idx = 0
    for y in range(0, H - kh + 1 + 2*pad, stride):
        for x0 in range(0, W - kw + 1 + 2*pad, stride):
            patch = cp.zeros((C, kh, kw), dtype=cp.float32)
            for c in range(C):
                for i in range(kh):
                    for j in range(kw):
                        yy = y + i - pad
                        xx = x0 + j - pad
                        if 0 <= yy < H and 0 <= xx < W:
                            patch[c, i, j] = x[c, yy, xx]
            cols[:, idx] = patch.reshape(-1)
            idx += 1
    return cols

def run_naive_conv(input_ch, input_h, input_w, kh, kw, stride=1, pad=0):
    # construct random input and kernel, single-output-channel for simplicity
    x = cp.random.random((input_ch, input_h, input_w), dtype=cp.float32)
    w = cp.random.random((1, input_ch, kh, kw), dtype=cp.float32)
    cols = im2col_gpu(x, kh, kw, stride=stride, pad=pad)  # shape (C*kh*kw, out_h*out_w)
    w_col = w.reshape(1, -1)  # (1, C*kh*kw)
    out = w_col.dot(cols)  # (1, out_h*out_w)
    return out

# cuDNN convolution via cupy
def run_cudnn_conv(x, w, stride=1, pad=0):
    # x: (N, C, H, W), w: (M, C, kh, kw)
    # Use cupy.nn.convolution via raw cudnn wrapper
    import cupyx
    return cupyx.nn.convolution(x, w, pad=(pad, pad), stride=(stride, stride))

# Utilities
def flops_gemm(M, N, K):
    return 2.0 * M * N * K

def print_result(name, seconds, flops, mem_free_before, mem_free_after):
    gflops = flops / seconds / 1e9
    mem_used_mb = (mem_free_before - mem_free_after) / (1024**2)
    print(f"{name}: {seconds:.6f}s, {gflops:.2f} GFLOPS, GPU mem delta {mem_used_mb:.2f} MB")

def benchmark_matmul(sizes=[256,512,1024]):
    print("MatMul benchmarks (float32):")
    for n in sizes:
        M = N = K = n
        A = cp.random.random((M,K), dtype=cp.float32)
        B = cp.random.random((K,N), dtype=cp.float32)
        C = cp.zeros((M,N), dtype=cp.float32)

        # warm
        run_cublas_mm(A,B,C)
        cp.cuda.Device().synchronize()

        free_before, total = cp.cuda.runtime.memGetInfo()
        t = time_kernel(lambda: run_naive_mm(A, B, C))
        fl = flops_gemm(M,N,K)
        free_after, _ = cp.cuda.runtime.memGetInfo()
        print_result(f"naive_mm {n}", t, fl, free_before, free_after)

        free_before, _ = cp.cuda.runtime.memGetInfo()
        t = time_kernel(lambda: run_tiled_mm(A, B, C))
        free_after, _ = cp.cuda.runtime.memGetInfo()
        print_result(f"tiled_mm {n}", t, fl, free_before, free_after)

        free_before, _ = cp.cuda.runtime.memGetInfo()
        t = time_kernel(lambda: run_cublas_mm(A, B, C))
        free_after, _ = cp.cuda.runtime.memGetInfo()
        print_result(f"cublas_mm {n}", t, fl, free_before, free_after)

def benchmark_conv():
    print("Conv benchmarks (simple single-output-channel naive vs cuDNN):")
    input_ch = 3
    H = W = 64
    kh = kw = 3
    # naive im2col
    free_before, _ = cp.cuda.runtime.memGetInfo()
    t0 = time.time()
    out = run_naive_conv(input_ch, H, W, kh, kw)
    cp.cuda.Device().synchronize()
    t1 = time.time()
    free_after, _ = cp.cuda.runtime.memGetInfo()
    # compute FLOPs roughly: output_elements * kernel_elements * 2
    out_h = H - kh + 1
    out_w = W - kw + 1
    fl = 2.0 * out_h * out_w * input_ch * kh * kw
    print_result("naive_conv_im2col", t1 - t0, fl, free_before, free_after)

    # cuDNN benchmark using cupy: build NCHW tensors
    x = cp.random.random((1, input_ch, H, W), dtype=cp.float32)
    w = cp.random.random((1, input_ch, kh, kw), dtype=cp.float32)
    free_before, _ = cp.cuda.runtime.memGetInfo()
    start = cp.cuda.Event(); end = cp.cuda.Event(); start.record()
    y = run_cudnn_conv(x, w)
    end.record(); end.synchronize()
    free_after, _ = cp.cuda.runtime.memGetInfo()
    t = cp.cuda.get_elapsed_time(start, end) * 1e-3
    print_result("cudnn_conv", t, fl, free_before, free_after)

if __name__ == '__main__':
    benchmark_matmul(sizes=[256,512])
    benchmark_conv()
