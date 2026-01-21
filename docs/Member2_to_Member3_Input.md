Member 2 → Member 3: Input Specification

Purpose
- Standardized JSON format that Member 2 (Environment) will send to Member 3 (GPU Programmer).
- Includes algorithm action sequence, matrix/test instances, constraints, reproducibility info, and optional simulated costs.

Top-level JSON fields (required)
- id: string — unique run identifier.
- env_version: string — environment tag/version.
- seed: integer — RNG seed for deterministic test generation.
- actions: [ action ] — ordered list of actions (see Action schema).
- matrix_specs: [ matrix_spec ] — list of matrices to use in the run.
- constraints: object — hardware/memory/precision constraints.
- reference_validation: object — how to obtain/calc reference outputs and tolerances.

Action schema (element of `actions`)
- type: string — one of e.g. "tile","loop_order","layout","prefetch","fuse_conv","transform".
- params: object — action-specific parameters. Examples:
  - tile: { "tile_m":64, "tile_n":64, "tile_k":16 }
  - loop_order: { "order":["i","k","j"] }
  - layout: { "A":"row","B":"col","C":"row" }
  - prefetch: { "levels":1, "buffer_bytes":8192 }

Matrix spec (element of `matrix_specs`)
- name: string — e.g., "A", "B", "C".
- shape: [int,int] — rows, cols (or [N,C,H,W] for conv if applicable).
- dtype: string — e.g., "float32","float16","bfloat16".
- generator: optional object — { "method":"random|identity|pattern", "params":{...} } or provide reference_seed.

Constraints (object)
- memory_limit_bytes: integer (optional)
- max_tile: integer (optional)
- alignment_bytes: integer (optional)
- allow_approx: boolean (if approximate algos allowed)

Reference validation (object)
- method: string — "numpy","scipy","provided".
- rtol: float
- atol: float
- reference_blob: optional path or base64-encoded small reference output (for small tests)

Optional fields
- reward_trace: [float] — per-step simulated reward values for debugging.
- simulated_costs: object — FLOP estimate, memory traffic estimate, predicted latency.
- hardware_hint: string — e.g., "A100","RTX6000" (helps Member 3 tune params).

Measurement protocol recommendation (include in JSON or follow by default)
- warmup_runs: 5
- timed_runs: 20
- report: median, mean, std

Minimal example (single-line summary)
{
  "id":"run-0001",
  "env_version":"v0.3.1",
  "seed":1234,
  "actions":[{"type":"tile","params":{"tile_m":64,"tile_n":64,"tile_k":16}},{"type":"loop_order","params":{"order":["i","k","j"]}}],
  "matrix_specs":[{"name":"A","shape":[512,512],"dtype":"float32"},{"name":"B","shape":[512,512],"dtype":"float32"}],
  "constraints":{"memory_limit_bytes":1073741824},
  "reference_validation":{"method":"numpy","rtol":1e-5,"atol":1e-6}
}

Notes for Member 3
- Member 3 must map `actions` → concrete kernel/block/grid/shared-memory config.
- If `reference_blob` not included, compute reference with CPU BLAS using provided seed and generator.
- Report any infeasible actions (e.g., tile exceeds memory) back to Member 2 as a short JSON error.