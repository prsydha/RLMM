"""
GPU Agent-to-Kernel Converter for 2x2 Matrix Multiplication
Converts agent-discovered Rank-1 tensor decompositions into GPU kernels
and benchmarks against naive and Strassen's algorithms.
"""

import time
import json
import numpy as np
import cupy as cp
from cupy import RawKernel
from typing import List, Dict, Tuple
import platform


# Naive 2x2 matrix multiplication kernel
NAIVE_KERNEL = RawKernel(r"""
extern "C" __global__
void matmul_naive_2x2(const float* A, const float* B, float* C) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    if (row < 2 && col < 2) {
        float sum = 0.0f;
        for (int k = 0; k < 2; k++) {
            sum += A[row * 2 + k] * B[k * 2 + col];
        }
        C[row * 2 + col] = sum;
    }
}
""", "matmul_naive_2x2")


# Strassen's 2x2 matrix multiplication kernel (7 multiplications)
STRASSEN_KERNEL = RawKernel(r"""
extern "C" __global__
void matmul_strassen_2x2(const float* A, const float* B, float* C) {
    int idx = threadIdx.x;
    
    if (idx < 7) {
        // Strassen's 7 multiplications
        // M1 = (A00 + A11)(B00 + B11)
        // M2 = (A10 + A11)B00
        // M3 = A00(B01 - B11)
        // M4 = A11(B10 - B00)
        // M5 = (A00 + A01)B11
        // M6 = (A10 - A00)(B00 + B01)
        // M7 = (A01 - A11)(B10 + B11)
        
        extern __shared__ float M[7];
        
        float a00 = A[0], a01 = A[1];
        float a10 = A[2], a11 = A[3];
        float b00 = B[0], b01 = B[1];
        float b10 = B[2], b11 = B[3];
        
        if (idx == 0) M[0] = (a00 + a11) * (b00 + b11);
        if (idx == 1) M[1] = (a10 + a11) * b00;
        if (idx == 2) M[2] = a00 * (b01 - b11);
        if (idx == 3) M[3] = a11 * (b10 - b00);
        if (idx == 4) M[4] = (a00 + a01) * b11;
        if (idx == 5) M[5] = (a10 - a00) * (b00 + b01);
        if (idx == 6) M[6] = (a01 - a11) * (b10 + b11);
        
        __syncthreads();
        
        // Combine to get result
        // C00 = M1 + M4 - M5 + M7
        // C01 = M3 + M5
        // C10 = M2 + M4
        // C11 = M1 - M2 + M3 + M6
        if (idx == 0) C[0] = M[0] + M[3] - M[4] + M[6];
        if (idx == 1) C[1] = M[2] + M[4];
        if (idx == 2) C[2] = M[1] + M[3];
        if (idx == 3) C[3] = M[0] - M[1] + M[2] + M[5];
    }
}
""", "matmul_strassen_2x2")


def parse_agent_output(algorithm_data: Dict) -> List[Dict[str, np.ndarray]]:
    """
    Parse agent output from gym environment into list of uvw tensors.
    
    Args:
        algorithm_data: Dictionary from env.get_algorithm_description()
                       containing 'rank1_tensors' list with u, v, w arrays
    
    Returns:
        List of dictionaries with numpy arrays for u, v, w
    """
    rank1_tensors = algorithm_data.get('rank1_tensors', [])
    uvw_list = []
    
    for tensor in rank1_tensors:
        uvw_list.append({
            'u': np.array(tensor['u'], dtype=np.float32),
            'v': np.array(tensor['v'], dtype=np.float32),
            'w': np.array(tensor['w'], dtype=np.float32)
        })
    
    return uvw_list


def generate_agent_kernel(uvw_list: List[Dict[str, np.ndarray]]) -> RawKernel:
    """
    Generate a GPU kernel from agent's uvw decomposition.
    
    For 2x2 matrices, each Rank-1 tensor contributes to the result.
    The formula: C = sum_i (u_i ⊗ v_i ⊗ w_i)
    
    Args:
        uvw_list: List of dictionaries containing u, v, w arrays
    
    Returns:
        Compiled CuPy RawKernel
    """
    num_tensors = len(uvw_list)
    
    # Generate kernel code
    kernel_code = f"""
extern "C" __global__
void matmul_agent_2x2(const float* A, const float* B, float* C) {{
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    if (row < 2 && col < 2) {{
        float result = 0.0f;
        
        // Agent discovered {num_tensors} rank-1 tensors
"""
    
    # Add each rank-1 tensor contribution
    for i, uvw in enumerate(uvw_list):
        u, v, w = uvw['u'], uvw['v'], uvw['w']
        
        kernel_code += f"""
        // Rank-1 tensor {i + 1}
        {{
            // Linear combination of A elements based on u
            float a_combo = 0.0f;
"""
        # u maps to input A matrix (flattened: a00, a01, a10, a11)
        for j in range(len(u)):
            if u[j] != 0:
                a_row, a_col = j // 2, j % 2
                kernel_code += f"            a_combo += {u[j]:.6f}f * A[{a_row} * 2 + {a_col}];\n"
        
        kernel_code += """
            // Linear combination of B elements based on v
            float b_combo = 0.0f;
"""
        # v maps to input B matrix
        for j in range(len(v)):
            if v[j] != 0:
                b_row, b_col = j // 2, j % 2
                kernel_code += f"            b_combo += {v[j]:.6f}f * B[{b_row} * 2 + {b_col}];\n"
        
        # w determines contribution to output C
        kernel_code += """
            // Contribution weight based on w
            float contribution = a_combo * b_combo;
"""
        
        # w maps to output C matrix
        for j in range(len(w)):
            if w[j] != 0:
                c_row, c_col = j // 2, j % 2
                kernel_code += f"""
            if (row == {c_row} && col == {c_col}) {{
                result += {w[j]:.6f}f * contribution;
            }}
"""
        
        kernel_code += "        }\n"
    
    kernel_code += """
        C[row * 2 + col] = result;
    }
}
"""
    
    return RawKernel(kernel_code, "matmul_agent_2x2")


def time_kernel(kernel, grid, block, args, shared_mem=0, warmup=5, iterations=20) -> float:
    """
    Time kernel execution with CUDA events.
    
    Args:
        kernel: CuPy RawKernel to benchmark
        grid: Grid dimensions
        block: Block dimensions
        args: Kernel arguments tuple
        shared_mem: Shared memory size in bytes
        warmup: Number of warmup runs
        iterations: Number of timed iterations
    
    Returns:
        Median latency in microseconds
    """
    # Warmup
    for _ in range(warmup):
        kernel(grid, block, args, shared_mem=shared_mem)
    cp.cuda.runtime.deviceSynchronize()
    
    # Timed runs
    latencies = []
    for _ in range(iterations):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        
        start.record()
        kernel(grid, block, args, shared_mem=shared_mem)
        end.record()
        end.synchronize()
        
        # Convert to microseconds
        latency_us = cp.cuda.get_elapsed_time(start, end) * 1000
        latencies.append(latency_us)
    
    return float(np.median(latencies))


def verify_correctness(C_test: cp.ndarray, C_ref: cp.ndarray) -> Dict:
    """
    Verify correctness of matrix multiplication result.
    
    Args:
        C_test: Test result matrix
        C_ref: Reference result matrix
    
    Returns:
        Dictionary with correctness metrics
    """
    diff = cp.abs(C_test - C_ref)
    max_abs_error = float(cp.max(diff))
    
    # Relative error where reference is non-zero
    mask = cp.abs(C_ref) > 1e-10
    if cp.any(mask):
        rel_error = diff[mask] / cp.abs(C_ref[mask])
        max_rel_error = float(cp.max(rel_error))
    else:
        max_rel_error = 0.0
    
    passed = max_abs_error < 1e-5
    
    return {
        "passed": passed,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "rtol": 1e-5,
        "atol": 1e-6
    }


def benchmark_implementation(
    kernel,
    kernel_name: str,
    source: str,
    A: cp.ndarray,
    B: cp.ndarray,
    C_ref: cp.ndarray,
    grid: Tuple,
    block: Tuple,
    shared_mem: int = 0,
    num_multiplications: int = 8,
    actions: List[Dict] = None
) -> Dict:
    """
    Benchmark a single implementation.
    
    Args:
        kernel: CuPy RawKernel
        kernel_name: Name of the kernel
        source: Source type (naive/strassen/agent)
        A, B: Input matrices
        C_ref: Reference result for correctness check
        grid, block: Launch parameters
        shared_mem: Shared memory size
        num_multiplications: Number of scalar multiplications
        actions: List of optimization actions taken
    
    Returns:
        Dictionary with benchmark results
    """
    C_test = cp.zeros((2, 2), dtype=cp.float32)
    
    # Run kernel
    latency_us = time_kernel(kernel, grid, block, (A, B, C_test), shared_mem)
    
    # Verify correctness
    correctness = verify_correctness(C_test, C_ref)
    
    # Build result
    result = {
        "name": source,
        "source": source,
        "kernel_name": kernel_name,
        "actions": actions or [],
        "correctness": correctness,
        "performance": {
            "latency_us": round(latency_us, 3),
            "num_multiplications": num_multiplications,
            "op_count": num_multiplications * 2  # mult + add per operation
        },
        "kernel": {
            "launch": {
                "grid": list(grid),
                "block": list(block),
                "shared_mem_bytes": shared_mem
            }
        }
    }
    
    return result


def run_all_benchmarks(uvw_list: List[Dict[str, np.ndarray]], 
                       matrix_size: Tuple[int, int] = (2, 2)) -> Dict:
    """
    Run all benchmarks: naive, Strassen's, and agent-discovered algorithm.
    
    Args:
        uvw_list: List of Rank-1 tensors from agent
        matrix_size: Matrix dimensions (default 2x2)
    
    Returns:
        Complete benchmark results in JSON format
    """
    m, n = matrix_size
    
    # Create test matrices
    A = cp.random.random((m, n), dtype=cp.float32)
    B = cp.random.random((m, n), dtype=cp.float32)
    
    # Reference result using cuBLAS
    C_ref = cp.dot(A, B)
    
    # GPU info
    device = cp.cuda.Device()
    gpu_name = device.compute_capability  # You can make this more detailed
    
    implementations = []
    
    # 1. Naive implementation
    print("Benchmarking naive algorithm...")
    naive_result = benchmark_implementation(
        NAIVE_KERNEL,
        "matmul_naive_2x2",
        "naive",
        A, B, C_ref,
        grid=(1, 1, 1),
        block=(2, 2, 1),
        num_multiplications=8,
        actions=[]
    )
    implementations.append(naive_result)
    
    # 2. Strassen's algorithm
    print("Benchmarking Strassen's algorithm...")
    strassen_result = benchmark_implementation(
        STRASSEN_KERNEL,
        "matmul_strassen_2x2",
        "strassen",
        A, B, C_ref,
        grid=(1, 1, 1),
        block=(7, 1, 1),
        shared_mem=7 * 4,  # 7 floats
        num_multiplications=7,
        actions=[{"type": "algorithm_optimization", "description": "Strassen's 7-multiplication algorithm"}]
    )
    implementations.append(strassen_result)
    
    # 3. Agent-discovered algorithm
    print(f"Benchmarking agent algorithm ({len(uvw_list)} rank-1 tensors)...")
    agent_kernel = generate_agent_kernel(uvw_list)
    agent_result = benchmark_implementation(
        agent_kernel,
        "matmul_agent_2x2",
        "agent",
        A, B, C_ref,
        grid=(1, 1, 1),
        block=(2, 2, 1),
        num_multiplications=len(uvw_list),
        actions=[{
            "type": "tensor_decomposition",
            "description": f"Agent-discovered {len(uvw_list)}-rank decomposition"
        }]
    )
    implementations.append(agent_result)
    
    # Build final JSON structure
    result = {
        "run_id": f"run-{int(time.time())}",
        "problem": {
            "matrix_shape": [m, n],
            "dtype": "float32",
            "operation": "matmul"
        },
        "environment": {
            "gpu": f"CUDA Device {device.id} (Compute {device.compute_capability})",
            "python": platform.python_version(),
            "cupy": cp.__version__
        },
        "implementations": implementations,
        "summary": {
            "best_latency": min(impl["performance"]["latency_us"] for impl in implementations),
            "best_multiplications": min(impl["performance"]["num_multiplications"] for impl in implementations),
            "all_correct": all(impl["correctness"]["passed"] for impl in implementations)
        }
    }
    
    return result


def main():
    """Main entry point for testing."""
    
    # Example: Strassen's algorithm as uvw decomposition
    # This is a simplified example - actual agent output from gym.py
    example_agent_output = {
        "matrix_size": {"m": 2, "n": 2, "p": 2},
        "num_multiplications": 7,
        "rank1_tensors": [
            {"u": [1, 0, 0, 1], "v": [1, 0, 0, 1], "w": [1, 0, 0, 1]},
            {"u": [0, 0, 1, 1], "v": [1, 0, 0, 0], "w": [0, 0, 1, -1]},
            {"u": [1, 0, 0, 0], "v": [0, 1, 0, -1], "w": [0, 1, 0, 1]},
            {"u": [0, 0, 0, 1], "v": [-1, 0, 1, 0], "w": [1, 0, 1, 0]},
            {"u": [1, 1, 0, 0], "v": [0, 0, 0, 1], "w": [-1, 1, 0, 0]},
            {"u": [-1, 0, 1, 0], "v": [1, 1, 0, 0], "w": [0, 0, 0, 1]},
            {"u": [0, 1, 0, -1], "v": [0, 0, 1, 1], "w": [1, 0, 0, 0]},
        ]
    }
    
    # Parse agent output
    uvw_list = parse_agent_output(example_agent_output)
    
    # Run benchmarks
    print("=" * 70)
    print("GPU Agent-to-Kernel Benchmark")
    print("=" * 70)
    results = run_all_benchmarks(uvw_list)
    
    # Print results
    print("\n" + json.dumps(results, indent=2))
    
    # Save to file
    output_file = "agent_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    print("=" * 70)
    
    # Print summary
    print("\nSummary:")
    for impl in results["implementations"]:
        status = "✅" if impl["correctness"]["passed"] else "❌"
        print(f"  {status} {impl['name']:10s}: {impl['performance']['latency_us']:8.3f} μs "
              f"({impl['performance']['num_multiplications']} mults)")


if __name__ == "__main__":
    main()
