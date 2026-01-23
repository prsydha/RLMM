import sys
import os
from pathlib import Path
import numpy as np
import subprocess

# Run benchmark visualization automatically
try:
    print("Running benchmark update with trained agent...")
    subprocess.call([sys.executable, "update_benchmark.py"])
    subprocess.call([sys.executable, "visualize_benchmark.py"])
except Exception as e:
    print(f"Failed to run benchmark update: {e}")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import torch

from env.gym import TensorDecompositionEnv
from agent.mcts_agent import MCTSAgent
from models.pv_network import PolicyValueNet
from project.logger import init_logger, log_metrics
from mlo.checkpoint import save_checkpoint

config = {
    "run_name": "alphatensor_mcts_v1",
    "matrix_size": (2, 2, 2),
    "max_rank": 20,
    "episodes": 200,
    "mcts_simulations": 50,
    "checkpoint_interval": 20,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

init_logger(config)

start_time = time.time()

# --------------------
# Initialize env, model, agent
# --------------------

env = TensorDecompositionEnv(
    matrix_size=config["matrix_size"],
    max_rank=config["max_rank"]
)

model = PolicyValueNet().to(config["device"])

agent = MCTSAgent(
    model=model,
    env=env,
    n_simulations=config["mcts_simulations"],
    device=config["device"]
)

episode_summaries = []

checkpoints_summary = []

# --------------------
# Training loop
# --------------------

global_step = 0

for episode in range(config["episodes"]):
    obs, info = env.reset()
    done = False
    episode_reward = 0
    step_count = 0

    start_episode_time = time.time()

    while not done:
        # # MCTS search (measure latency)
        # t0 = time.time()
        u, v, w = agent.search(obs)
        action = {
            'u': u,
            'v': v,
            'w': w
        }

        # search_latency_us = (time.time() - t0) * 1e6

        # Environment step
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print("\n\n Done status: ", done)

        episode_reward += reward
        step_count += 1
        global_step += 1

        # --------------------
        # Action sparsity computation
        # --------------------
        # u, v, w = env._action_to_rank1_tensor(action)

        action_sparsity = (
            np.sum(u == 0) +
            np.sum(v == 0) +
            np.sum(w == 0)
        ) / (len(u) + len(v) + len(w))

        # --------------------
        # Log step metrics
        # --------------------
        log_metrics({
            "step_reward": reward,
            # "search_latency_us": search_latency_us,
            "residual_norm": info["residual_norm"],
            "rank_used": info["rank_used"],
            "action_valid": int(info["action_valid"]),
            "action_sparsity": action_sparsity,
            "action": action
        }, step=global_step)

        print(
        f"Episode {episode:03d} | "
        f"Reward: {episode_reward:7.2f} | "
        f"Rank: {info['rank_used']} | "
        f"Residual: {info['residual_norm']:.4e}"
        f"Action: {action} | "
        f"Steps: {step_count} | "
        f"Observation residual: {np.linalg.norm(obs):.2f} | "
    
    )

    # --------------------
    # Episode summary table
    # --------------------
    episode_summaries.append((episode, episode_reward, info["residual_norm"], info["rank_used"]))
    try:
        import wandb
        table = wandb.Table(columns=["Episode", "Reward", "Residual", "Rank"])
        for ep, rew, res, rnk in episode_summaries:
            table.add_data(ep, rew, res, rnk)
        wandb.log({"episode_summary": table}, step=global_step)
    except Exception as e:
        print(f"WandB logging failed: {e}. Skipping episode summary log.")
    # print(episode, episode_reward, info["residual_norm"], info["rank_used"])

    # --------------------
    # Checkpoint
    # --------------------
    if episode % 5 == 0:  # Reduced interval to 5 for better visibility
        save_checkpoint(model, global_step, start_time=start_time, episode=episode)
        checkpoints_summary.append({
            "episode": episode, 
            "step": global_step, 
            "run_duration": time.time() - start_time, 
            "file": f"model_step_{global_step}.pt"
        })
        try:
            import wandb
            table = wandb.Table(columns=["Episode", "Step", "Run Duration (s)", "File"])
            for ckpt in checkpoints_summary:
                table.add_data(ckpt["episode"], ckpt["step"], ckpt["run_duration"], ckpt["file"])
            wandb.log({"checkpoints_summary": table}, step=global_step)
        except Exception as e:
            print(f"WandB checkpoint table logging failed: {e}")

    # print(
    #     f"Episode {episode:03d} | "
    #     f"Reward: {episode_reward:7.2f} | "
    #     f"Rank: {info['rank_used']} | "
    #     f"Residual: {info['residual_norm']:.4e}"
    #     f"Action: {action} | "
    #     f"Steps: {step_count} | "
    #     f"Observation residual: {np.linalg.norm(obs):.2f} | "
    
    # )

# Log total run time
total_time = time.time() - start_time
log_metrics({"total_run_time": total_time}, step=config["episodes"])
print(f"Training completed in {total_time:.2f} seconds")

# Update benchmark with trained agent
try:
    print("Updating benchmark with trained agent...")
    subprocess.call([sys.executable, "update_benchmark.py"])
    subprocess.call([sys.executable, "visualize_benchmark.py"])
except Exception as e:
    print(f"Failed to update benchmark: {e}")
