Member 3 → Member 4: Output Specification

Purpose
- Standardized outputs and artifacts that Member 3 will hand to Member 4 for MLOps, benchmarking and reporting.

Top-level JSON fields (required)
- id: string — same run id received from Member 2.
- actions: [action] — copy of the input `actions` plus `mapped_params` used by the kernel.
- matrix_specs: [matrix_spec] — copy of inputs tested.
- kernel_artifact: object — pointers to kernel source and build artifacts.
- correctness: object — numeric-error metrics and pass/fail.
- performance: object — timing and throughput metrics.
- profiling: object — paths to raw profiler outputs and a short summary.
- reproducibility: object — environment (GPU model, CUDA version, driver), build flags, commands.

kernel_artifact (object)
- kernel_name: string
- source_path: string (relative path)
- build_command: string
- launch_params: { "grid": [gx,gy,gz], "block": [bx,by,bz], "shared_mem_bytes": int }

correctness (object)
- passed: boolean
- rtol: float
- atol: float
- max_abs_error: float
- max_rel_error: float
- reference_method: string
- sample_diffs_path: optional path to small arrays showing differences

performance (object)
- warmup_runs: int
- timed_runs: int
- latency_ms_median: float
- latency_ms_mean: float
- latency_ms_std: float
- GFLOPS_median: float
- bytes_moved_estimate: integer
- occupancy_estimate: float (0-1)

profiling (object)
- nvprof_path | nsight_path: string (where raw trace is saved)
- sm_efficiency: float (if available)
- l2_hit_rate: float (if available)
- brief_summary: string

reproducibility (object)
- gpu_model: string
- cuda_version: string
- driver_version: string
- compiler: string
- container_image: optional string (docker image tag)
- command_line: the exact command used to run the benchmark

Outputs & files to include (organize under `results/<id>/`)
- `kernel.cu` or `kernel.py` (Triton) — kernel source
- `build.log` — build output
- `results.json` — full JSON conforming to this schema
- `profile/` — raw profiler outputs
- `correctness/` — small reference and diff arrays for small tests
- `plots/` — latency/GFLOPS PNGs (optional)

CSV summary line (recommended)
- id, matrix_shape, dtype, kernel_name, latency_ms_median, GFLOPS_median, max_abs_error, passed

Example `results.json` snippet
{
  "id":"run-0001",
  "actions":[{"type":"tile","params":{"tile_m":64,...},"mapped_params":{"block": [32,8,1],"shared_mem_bytes":12288}}],
  "matrix_specs":[{"name":"A","shape":[512,512],"dtype":"float32"}],
  "kernel_artifact":{"kernel_name":"gemm_tile_64","source_path":"results/run-0001/kernel.cu","build_command":"nvcc -O3 ...","launch_params":{"grid":[16,16,1],"block":[32,8,1],"shared_mem_bytes":12288}},
  "correctness":{"passed":true,"rtol":1e-5,"atol":1e-6,"max_abs_error":1.2e-6},
  "performance":{"warmup_runs":5,"timed_runs":20,"latency_ms_median":1.23,"GFLOPS_median":420.1},
  "profiling":{"nvprof_path":"results/run-0001/profile/nvprof.csv","brief_summary":"SM eff 78%"},
  "reproducibility":{"gpu_model":"A100","cuda_version":"12.0","container_image":"myimage:latest","command_line":"python run_benchmark.py results/input/run-0001.json"}
}

Notes for Member 4
- Member 4 should be able to ingest `results/<id>/results.json` and the `kernel_artifact.source_path` to reproduce benchmarks.
- Member 3 must supply the exact command(s) to re-run each benchmark and the Docker/conda environment or a `requirements.txt`/`environment.yml`.