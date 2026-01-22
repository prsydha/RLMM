import sys
import os
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import torch

from env.gym import TensorDecompositionEnv
from agent.mcts_agent import MCTSAgent
from models.pv_network import PolicyValueNet
from project.logger import init_logger, log_metrics
from mlo.checkpoints.checkpoint import save_checkpoint

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
    import wandb
    table = wandb.Table(columns=["Episode", "Reward", "Residual", "Rank"])
    table.add_data(
        episode,
        episode_reward,
        info["residual_norm"],
        info["rank_used"]
    )
    wandb.log({"episode_summary": table})
    # print(episode, episode_reward, info["residual_norm"], info["rank_used"])

    # --------------------
    # Checkpoint
    # --------------------
    if episode % config["checkpoint_interval"] == 0:
        save_checkpoint(model, episode)

    # print(
    #     f"Episode {episode:03d} | "
    #     f"Reward: {episode_reward:7.2f} | "
    #     f"Rank: {info['rank_used']} | "
    #     f"Residual: {info['residual_norm']:.4e}"
    #     f"Action: {action} | "
    #     f"Steps: {step_count} | "
    #     f"Observation residual: {np.linalg.norm(obs):.2f} | "
    
    # )
