GPU Baselines (kernels/gpu)

This folder contains a CuPy-based benchmarking harness implementing GPU baselines:

- `benchmark.py`: runs naive and tiled matmul (RawKernel), cuBLAS via CuPy, naive convolution via im2col+matmul, and cuDNN convolution via CuPy.

Requirements
- Python 3.8+
- cupy matching your CUDA toolkit (e.g., `pip install cupy-cuda12x`)

Run
```
python kernels/gpu/benchmark.py
```

Notes
- The harness uses CuPy RawKernel to embed small CUDA kernels directly in Python.
- It measures execution time with CUDA events and reports GFLOPS and GPU memory delta.
