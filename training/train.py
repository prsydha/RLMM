import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.gym import TensorDecompositionEnv
from agent.mcts_agent import MCTSAgent
from models.pv_network import PolicyValueNet
from project.logger import init_logger, log_metrics
# from mlo.checkpoints.checkpoint import save_checkpoint

# --- Hyperparameters ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 5000
EPOCHS = 100         # Total training loops
EPISODES_PER_EPOCH = 5  # Self-play games per loop
MCTS_SIMS = 50       # Search depth per move

config = {
    "run_name": "alphatensor_mcts_v1",
    "matrix_size": (2, 2, 2),
    "max_rank": 10,
    "learning rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "replay_buffer_size": REPLAY_BUFFER_SIZE,
    "epochs": EPOCHS,
    "episodes_per_epoch": EPISODES_PER_EPOCH,
    "mcts_simulations": 50,
    "checkpoint_interval": 20,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def train():
    # 1. setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # initialize Environment, Network and Optimizer
    env = TensorDecompositionEnv(matrix_size=config["matrix_size"], max_rank=config["max_rank"])
    net = PolicyValueNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Replay Buffer: Stores (state, mcts_probs, winner)
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE) # use deque for efficient pops from left(front)

    # --------------------
    # 2. Main Training Loop
    # --------------------

    init_logger(config)

    global_step = 0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

        # --- A. Self-Play Phase (Data Collection) ---
        net.eval() # Set to evaluation mode for inference

        for episode in range(EPISODES_PER_EPOCH):
            obs, info = env.reset()
            mcts = MCTSAgent(model = net, env = env, n_simulations=MCTS_SIMS, device=device)

            episode_data = [] # Stores (state, action_dist) for this game
            steps = 0
            episode_reward = 0
            done = False

            # play one full game
            while not done:
                # run MCTS to get the best action distribution
                # we need MCTS to return action probabilities, not just the best move
                # we'll peek into the root node of the MCTS after search
                best_action = mcts.search(obs)
            
                # extract visit counts from Root -> Policy target
                # we need to map visit counts to the output heads
                # for simplicity, we train on the chosen action only
                # ideally, we would train on the full distribution vector

                # execute move (environment step)
                u, v, w = mcts._parse_action(best_action)
                action = {
                    'u': u,
                    'v': v,
                    'w': w
                }
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # store data : (state, action_tuple, result_placeholder)
                episode_data.append([obs.flatten(), best_action])

                obs = next_obs
                steps += 1
                episode_reward += reward
                global_step += 1

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
                "residual_norm": info["residual_norm"],
                "rank_used": info["rank_used"],
                "action_valid": int(info["action_valid"]),
                "action_sparsity": action_sparsity,
                "action": action
                }, step=global_step)

                print(
                f"Episode {episode:03d} | "
                f"Reward: {reward:7.2f} | "
                f"Rank: {info['rank_used']} | "
                f"Residual: {info['residual_norm']:.4e}"
                f"Action: {action} | "
                f"Steps: {steps} | "
                f"Observation residual: {np.linalg.norm(obs):.2f} | "
                )
                
            # end of one episode

            # --------------------
            # Episode summary table
            # --------------------
            import wandb
            table = wandb.Table(columns=["Episode", "Reward", "Residual", "Rank"])
            table.add_data(
                episode,
                reward,
                info["residual_norm"],
                info["rank_used"]
            )
            wandb.log({"episode_summary": table})

            # assign value (winner) to all steps in this episode
            # if solved, value =1 else -1
            final_value = 1.0 if terminated else -1.0
            print(f"  Episode {episode+1}: Steps={steps}, Result={final_value}")

            for data in episode_data:
                state_flat, action_tuple = data
                replay_buffer.append((state_flat, action_tuple, final_value))
            
        # --- B. Training Phase (Network Update)--- once per Epoch ( after several episodes )

        net.train() # set to training mode
        if len(replay_buffer) < BATCH_SIZE:
            print("Not enough data in replay buffer to train.")
            continue

        # run a few updates on the buffer
        total_loss = 0
        for _ in range(10): # 10 gradient steps per epoch
            batch = random.sample(replay_buffer, BATCH_SIZE)
            states, actions, values = zip(*batch) # unpacking batch and pairing up first elements, second elements, etc. together as lists

            # prepare tensors
            states_tensor = torch.FloatTensor(np.array(states)).to(device) # shape (BATCH_SIZE, state_dim) # pytorch converts numpy array to tensor faster and more reliably than raw Python lists
            values_tensor = torch.FloatTensor(np.array(values)).unsqueeze(1).to(device) # shape (BATCH_SIZE, 1)

            # forward pass
            policy_logits, value_pred = net(states_tensor)

            # We need to train the network on two losses:
            # Policy Loss: The network's "intuition" (probabilities) should match the "reality" (MCTS visit counts).
            # Value Loss: The network's predicted score should match the final result (Solved/Not Solved).
            
            # 1. Value Loss (MSE) : did we predict the win/loss correctly?
            value_loss = F.mse_loss(value_pred, values_tensor) # functional mse loss

            # 2. Policy Loss (cross-entropy) : did we pick the right integers?
            # we must split the 'actions' tuple (batch of 12/27 ints) into separate targets for each head
            # actions is a list of tuples: [(1, 0, -1, ...), (...), ...]
            # convert to tensor : (batch_size, 12/27)
            action_targets = torch.LongTensor(actions).to(device) # LongTensor for integer targets
            # map {-1, 0, 1} to {0, 1, 2} for cross-entropy
            action_targets = action_targets + 1

            policy_loss = 0
            # iterate over each head
            for i in range(len(policy_logits)):
                # head i output: shape (batch_size, 3)
                # target i : (batch_size) -> the i-th integer of the action tuple
                head_loss = F.cross_entropy(policy_logits[i], action_targets[:, i])
                policy_loss += head_loss
            
            # combine losses
            loss = value_loss + policy_loss

            optimizer.zero_grad() # clear previous gradients becuase by default, PyTorch accumulates gradients
            loss.backward()       # backpropagate to compute gradients
            optimizer.step()      # update weights

            total_loss += loss.item() # accumulate loss for logging

            # The variable "loss" is not just a number; it is a PyTorch Tensor that sits at the top of a complex computational graph. It carries "history" (the grad_fn) so that PyTorch knows how to backpropagate through it.

            # The .item() method extracts the raw Python scalar (a standard float) from the tensor.
            # It unwraps the value from the Tensor object.
            # It disconnects it from the computational graph.
            # It moves the value from GPU memory back to CPU memory.

        avg_loss = total_loss / 10 #
        print(f"  Average Loss: {avg_loss:.4f}")

        # Save Checkpoint
        if (epoch+1) % 10 == 0:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            torch.save(net.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()

