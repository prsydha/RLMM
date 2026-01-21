import time
import numpy as np
import torch

from env import TensorDecompositionEnv
from agent import MCTSAgent
from model import PolicyValueNet
from logger import init_logger,log_metrics
from checkpoint import save_checkpoint

config ={
    "run_name":"alphatensor_mcts_v1",
    "matrix_size":(2,2,2),
    "max_rank":20,
    "episodes":200,
    "mcts_simulations":50,
    "checkpoint_interval":20,
    "device":"cuda" if torch.cuda.is_available() else "cpu"
}

init_logger(config)

#init

env = TensorDecompositionEnv(
    matrix_size=config["matrix_size"],
    max_rank=onfig["max_rank"]
)

model = PolicyValueNet().to(config["device"])
agent= MCTSAgent(
    model=model,
    env=env,
    n_simulations=config["mcts_simulations"],
    device=config["device"]
)

#training loop

global_step =0

for episode in range(config["episodes"]):
    obs,info = env.reset()
    done = False
    episode_reward =0
    step_count =0
    
    start_episode_time = time.time()

    while not done:
        #mcts search (measure latency)
        t0 = time.time()
        action = agent.search(obs)
        search_latency_us = (time.time() - t0) * 1e6
        
        #env step
        
        obs,reward,terminated,truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward
        step_count +=1
        global_step +=1
        
        
        #log step metrics
        log_metrics({
            "step_reward":reward,
            "search_latency_us":search_latency_us,
            "residual_norm":info["residual norm"],
            "rank_used": info["rank_used"],
            "action_valid": int(info["action_valid"]),
            "action_sparsity": action.count(0),
        }, step=global_step)
        
    # Episode finished → log summary table
    import wandb
    table = wandb.Table(columns=["Episode", "Reward", "Residual", "Rank"])
    table.add_data(episode, episode_reward, info["residual_norm"], info["rank_used"])
    wandb.log({"episode_summary": table})    
        
    #checkpoint
    
    if episode % config["checkpoint_interval"] ==0:
        save_checkpoint(model, episode)
        
    print(
        f"Episode {episode:03d} |"
        f"Reward: {episode_reward:7.2f} | "
        f"Rank: {info['rank_used']} | "
        f"Residual: {info['residual_norm']:.4e}"  
    )