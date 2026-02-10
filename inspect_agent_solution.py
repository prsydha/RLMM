import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from env.gym import TensorDecompositionEnv
from agent.mcts_agent import MCTSAgent
from models.resnet_pv_network import PolicyValueNet
import config as project_config

def inspect_agent():
    print("="*60)
    print("AGENT SOLUTION INSPECTOR")
    print("="*60)

    # 1. Setup Environment
    env = TensorDecompositionEnv(
        matrix_size=(2, 2, 2),
        max_rank=20
    )
    
    # 2. Load Model
    device = "cpu"
    checkpoint_path = "checkpoints/latest_model.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = PolicyValueNet(
        input_dim=project_config.INPUT_DIM,
        hidden_dim=project_config.HIDDEN_DIM,
        n_heads=project_config.N_HEADS
    ).to(device)
    
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # 3. Initialize Agent
    agent = MCTSAgent(
        model=model,
        env=env,
        n_simulations=100, # Sufficient for inspection
        device=device,
        temperature=0.01  # Greedy as possible
    )

    # 4. Run One Episode and Print Factors
    print("\nStarting evaluation episode...")
    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0
    
    initial_norm = info["residual_norm"]
    print(f"Initial Residual Norm: {initial_norm:.4f}")
    
    while not done and step < 20:
        step += 1
        print(f"\n--- STEP {step} ---")
        
        # Get action from agent
        action_tuple = agent.search(obs, add_noise=False)
        u, v, w = agent._parse_action(action_tuple)
        
        # Display the factors
        print(f"u: {u.tolist()}")
        print(f"v: {v.tolist()}")
        print(f"w: {w.tolist()}")
        
        # Take step in environment
        action = {'u': u, 'v': v, 'w': w}
        obs, reward, terminated, truncated, info = env.step(action)
        
        curr_norm = info["residual_norm"]
        print(f"Step Result -> Norm: {curr_norm:.4f}, Valid: {info['action_valid']}, Reward: {reward:.4f}")
        
        done = terminated or truncated
        
        if terminated:
            print(f"\n✅ SOLVED in {step} steps!")
            break

    if not terminated:
        print(f"\n❌ FAILED to solve in {step} steps. Final Residual Norm: {info['residual_norm']:.4f}")

if __name__ == "__main__":
    inspect_agent()
