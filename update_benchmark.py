import sys
import os
from pathlib import Path
import numpy as np
import torch
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.gym import TensorDecompositionEnv
from agent.mcts_agent import MCTSAgent
from models.pv_network import PolicyValueNet
from mlo.checkpoint import load_checkpoint

# Fix for CuPy/CUDA DLL error on Windows
import os
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin"
if not os.path.exists(cuda_path):
    os.environ["CUPY_SKIP_CUDA_CHECK"] = "1"
    print(f"⚠️ Warning: CUDA bin path not found at {cuda_path}. Skipping CuPy DLL auto-load.")

try:
    from kernels.gpu.agent_to_kernel import run_all_benchmarks, parse_agent_output
except ImportError as e:
    print(f"Skipping benchmark update: {e}")
    print("Ensure 'cupy' is installed and a GPU is available.")
    sys.exit(0)

def update_benchmark_with_agent():
    # Load the latest checkpoint
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        print("No checkpoints found. Run training first.")
        return

    # Find the latest checkpoint
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("model_step_") and f.endswith(".pt")]
    if not checkpoints:
        print("No checkpoints found.")
        return

    latest = max(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    ckpt_path = os.path.join(checkpoint_dir, latest)
    step = int(latest.split("_")[-1].split(".")[0])

    print(f"Loading checkpoint: {ckpt_path} at step {step}")

    # Initialize env, model, agent
    config = {
        "matrix_size": (2, 2, 2),
        "max_rank": 20,
        "device": "cpu"
    }

    env = TensorDecompositionEnv(
        matrix_size=config["matrix_size"],
        max_rank=config["max_rank"]
    )

    model = PolicyValueNet().to(config["device"])
    load_checkpoint(model, ckpt_path)

    agent = MCTSAgent(
        model=model,
        env=env,
        n_simulations=50,  # Use same as training
        device=config["device"]
    )

    # Run agent to get decomposition
    obs, info = env.reset()
    done = False
    actions = []

    while not done:
        u, v, w = agent.search(obs)
        action = {
            'u': u,
            'v': v,
            'w': w
        }
        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Get the algorithm description
    algorithm_data = env.get_algorithm_description()
    print(f"Agent found decomposition with {len(algorithm_data['rank1_tensors'])} multiplications")

    # Parse to uvw_list
    uvw_list = parse_agent_output(algorithm_data)

    # Run benchmarks
    print("Running benchmarks...")
    results = run_all_benchmarks(uvw_list)

    # Save to file
    output_file = "agent_benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Updated benchmark results saved to {output_file}")

    # Print summary
    print("\nSummary:")
    for impl in results["implementations"]:
        status = "✅" if impl["correctness"]["passed"] else "❌"
        print(f"  {status} {impl['name']:10s}: {impl['performance']['latency_us']:8.3f} μs "
              f"({impl['performance']['num_multiplications']} mults)")

if __name__ == "__main__":
    update_benchmark_with_agent()