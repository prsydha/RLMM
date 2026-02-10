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
from models.resnet_pv_network import PolicyValueNet
# from mlo.checkpoint import load_checkpoint

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

    # Try to load latest_model.pth first (Production standard)
    latest_pth = os.path.join(checkpoint_dir, "latest_model.pth")
    
    if os.path.exists(latest_pth):
        ckpt_path = latest_pth
        print(f"Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        step = checkpoint.get("global_step", 0)
    else:
        # Fallback to step-based naming
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("model_step_") and f.endswith(".pt")]
        if not checkpoints:
            print("No checkpoints found.")
            return

        latest = max(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        ckpt_path = os.path.join(checkpoint_dir, latest)
        step = int(latest.split("_")[-1].split(".")[0])
        print(f"Loading checkpoint: {ckpt_path} at step {step}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")

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
    
    # Load state dict robustly
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)  # Direct state dict
    else:
        print("⚠️ Checkpoint format unknown!")
        model.load_state_dict(checkpoint) # Try anyway

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
        # Get the action tuple from MCTS
        action_tuple = agent.search(obs)
        
        # Parse the 12-tuple into u, v, w vectors (length 4 each for 2x2)
        u, v, w = agent._parse_action(action_tuple)
        
        action = {
            'u': u,
            'v': v,
            'w': w
        }
        actions.append(action)

        print(f"  Step {len(actions):02d}: Action Parsed")
        print(f"    u: {u.tolist()}")
        print(f"    v: {v.tolist()}")
        print(f"    w: {w.tolist()}")

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"    Result -> Norm: {info['residual_norm']:.4f}, Valid: {info['action_valid']}")
        
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