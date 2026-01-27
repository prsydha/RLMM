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

from env.gym import TensorDecompositionEnv
from agent.mcts_agent import MCTSAgent
from models.pv_network import PolicyValueNet
from project.logger import init_logger, log_metrics
from training.visualizer_server import VisualizerServer
# from mlo.checkpoint import save_checkpoint

# --- Hyperparameters ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 5000
EPOCHS = 100         # Total training loops
EPISODES_PER_EPOCH = 5  # Self-play games per loop
MCTS_SIMS = 75       # MCTS simulations per move (25-50: fast, 50-100: balanced, 100-200: quality, 200+: slow but strong)

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
    "checkpoint_interval": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# Start visualizer server
viz_server = VisualizerServer()
viz_server.start()

start_train_time = time.time()


def train():
    # 1. setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        torch.cuda.empty_cache()  # Clear cache before training
    else:
        print("⚠️ WARNING: Running on CPU - training will be MUCH slower!")
        print("   If you have a GPU, make sure CUDA is properly installed.")

    # initialize Environment, Network and Optimizer
    env = TensorDecompositionEnv(matrix_size=config["matrix_size"], max_rank=config["max_rank"])
    net = PolicyValueNet().to(device)
    
    # Initialize network weights with smaller values for better initial exploration
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    net.apply(init_weights)
    
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # Replay Buffer: Stores (state, mcts_probs, winner, priority)
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE) # use deque for efficient pops from left(front)
    
    # Track best performance
    best_rank_found = config["max_rank"]
    epochs_without_improvement = 0
    total_successes = 0
    recent_success_rate = 0.0

    # --------------------
    # 2. Main Training Loop
    # --------------------

    init_logger(config)

    global_step = 0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # Calculate exploration parameters
        temperature = max(0.1, 1.0 - (epoch / EPOCHS) * 0.9)
        epsilon = max(0.05, 0.5 - (epoch / EPOCHS) * 0.45)
        
        # Adaptive exploration: increase if not finding solutions
        if recent_success_rate < 0.1 and epoch > 10:
            epsilon = min(0.5, epsilon + 0.2)
            print(f"⚠️ Low success rate ({recent_success_rate:.1%}), increasing exploration")
        
        print(f"Temperature: {temperature:.3f}, Epsilon: {epsilon:.3f}, Success Rate: {recent_success_rate:.1%}")

        # --- A. Self-Play Phase (Data Collection) ---
        net.eval() # Set to evaluation mode for inference
        
        epoch_successes = 0

        for episode in range(EPISODES_PER_EPOCH):
            obs, info = env.reset()
            # Progressive MCTS: start with fewer simulations, increase over time
            current_sims = int(MCTS_SIMS * (0.5 + 0.5 * (epoch / EPOCHS)))
            current_sims = max(25, min(current_sims, MCTS_SIMS))
            mcts = MCTSAgent(model = net, env = env, n_simulations=current_sims, device=device, temperature=temperature)

            episode_data = [] # Stores (state, action_dist) for this game
            steps = 0
            episode_reward = 0
            valid_actions = 0
            invalid_actions = 0
            done = False

            # play one full game
            while not done:
                # Early exploration: use random actions sometimes
                epsilon = max(0.05, 0.5 - (epoch / EPOCHS) * 0.45)
                if np.random.random() < epsilon:
                    # Heuristic: prefer sparse actions (1-3 non-zero per vector)
                    def sparse_vector(size):
                        vec = np.zeros(size, dtype=int)
                        num_nonzero = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
                        positions = np.random.choice(size, size=min(num_nonzero, size), replace=False)
                        for pos in positions:
                            vec[pos] = np.random.choice([-1, 1])
                        return vec
                    
                    u = sparse_vector(4)
                    v = sparse_vector(4)
                    w = sparse_vector(4)
                    action = {'u': u, 'v': v, 'w': w}
                    best_action = tuple(list(u) + list(v) + list(w))
                else:
                    # run MCTS to get the best action distribution
                    # Add noise for exploration in early epochs
                    add_noise = epoch < EPOCHS // 2
                    best_action = mcts.search(obs, add_noise=add_noise)
                    u, v, w = mcts._parse_action(best_action)
                    action = {'u': u, 'v': v, 'w': w}
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # store data : (state, action_tuple, result_placeholder)
                episode_data.append([obs.flatten(), best_action])

                obs = next_obs
                steps += 1
                episode_reward += reward
                global_step += 1
                
                # Track valid/invalid actions
                if info["action_valid"]:
                    valid_actions += 1
                else:
                    invalid_actions += 1

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

                # Only print every 5th step to reduce clutter
                if steps % 5 == 0 or done:
                    print(
                    f"Ep {episode:03d} | "
                    f"R: {reward:6.1f} | "
                    f"Rank: {info['rank_used']} | "
                    f"Norm: {info['residual_norm']:.2e} | "
                    f"Step: {steps}"
                    )


                        # Broadcast to visualizer
                viz_server.broadcast({
                    "type": "step",
                    "global_step": global_step,
                    "episode": episode,
                    "step_count": steps,
                    "reward": reward,
                    "residual": info["residual_norm"],
                    "rank": info["rank_used"],
                    "sparsity": action_sparsity,
                    "action": action,
                    "elapsed": time.time() - start_train_time
                })
                        
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
            
            if terminated:
                epoch_successes += 1
                total_successes += 1
                print(f"  🎯 Episode {episode+1}: SUCCESS! Steps={steps}, Rank={env.algorithm.__len__()}, Valid={valid_actions}, Invalid={invalid_actions}, Reward={episode_reward:.2f}")
            else:
                print(f"  Episode {episode+1}: Steps={steps}, Valid={valid_actions}, Invalid={invalid_actions}, Result={final_value:.1f}, Reward={episode_reward:.2f}, FinalNorm={info['residual_norm']:.4e}")

            # Prioritize successful episodes and progress
            priority = 1.0 if terminated else 0.5
            if terminated and len(env.algorithm) < best_rank_found:
                priority = 2.0  # High priority for new best solutions
                best_rank_found = len(env.algorithm)
                
            for data in episode_data:
                state_flat, action_tuple = data
                replay_buffer.append((state_flat, action_tuple, final_value, priority))
        
        # Update success rate (exponential moving average)
        epoch_success_rate = epoch_successes / EPISODES_PER_EPOCH
        recent_success_rate = 0.7 * recent_success_rate + 0.3 * epoch_success_rate
        
        print(f"\n  Epoch Summary: {epoch_successes}/{EPISODES_PER_EPOCH} successful, Total successes: {total_successes}")
            
        # --- B. Training Phase (Network Update)--- once per Epoch ( after several episodes )

        net.train() # set to training mode
        if len(replay_buffer) < BATCH_SIZE:
            print("Not enough data in replay buffer to train.")
            continue

        # run a few updates on the buffer
        total_loss = 0
        total_value_loss = 0
        total_policy_loss = 0
        
        for _ in range(10): # 10 gradient steps per epoch
            # Prioritized sampling
            if len(replay_buffer) >= BATCH_SIZE:
                priorities = np.array([item[3] if len(item) > 3 else 1.0 for item in replay_buffer])
                priorities = priorities / priorities.sum()
                indices = np.random.choice(len(replay_buffer), BATCH_SIZE, p=priorities)
                batch = [replay_buffer[i] for i in indices]
            else:
                batch = list(replay_buffer)
            
            # Handle both old format (3 items) and new format (4 items)
            states, actions, values = [], [], []
            for item in batch:
                states.append(item[0])
                actions.append(item[1])
                values.append(item[2])

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

            optimizer.zero_grad() # clear previous gradients
            loss.backward()       # backpropagate to compute gradients
            
            # Gradient clipping for stability and track gradient norms
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
            optimizer.step()      # update weights
            
            # Track if gradients are vanishing/exploding
            if _ == 0:  # Only for first batch per epoch
                total_grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

            total_loss += loss.item() # accumulate loss for logging
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()

            # The variable "loss" is not just a number; it is a PyTorch Tensor that sits at the top of a complex computational graph. It carries "history" (the grad_fn) so that PyTorch knows how to backpropagate through it.

            # The .item() method extracts the raw Python scalar (a standard float) from the tensor.
            # It unwraps the value from the Tensor object.
            # It disconnects it from the computational graph.
            # It moves the value from GPU memory back to CPU memory.

        avg_loss = total_loss / 10
        avg_value_loss = total_value_loss / 10
        avg_policy_loss = total_policy_loss / 10
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"  Average Loss: {avg_loss:.4f} (Value: {avg_value_loss:.4f}, Policy: {avg_policy_loss:.4f})")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # GPU memory usage
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1e9
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"  GPU Memory: {gpu_mem_allocated:.2f}GB allocated, {gpu_mem_reserved:.2f}GB reserved")
        
        # Show progress estimate
        if total_successes > 0:
            avg_epochs_per_success = (epoch + 1) / total_successes
            print(f"  📈 Average epochs per success: {avg_epochs_per_success:.1f}")
        else:
            print(f"  ⏳ No successes yet. Keep training...")
        
        # Log epoch metrics
        log_metrics({
            "epoch_loss": avg_loss,
            "epoch_value_loss": avg_value_loss,
            "epoch_policy_loss": avg_policy_loss,
            "learning_rate": current_lr,
            "replay_buffer_size": len(replay_buffer),
            "best_rank_found": best_rank_found,
            "temperature": temperature,
            "epsilon": epsilon,
            "epoch_successes": epoch_successes,
            "total_successes": total_successes,
            "success_rate": recent_success_rate
        }, step=epoch)
        
        # Step learning rate scheduler
        scheduler.step()

        # Evaluation phase (no exploration)
        if (epoch+1) % 10 == 0:
            print(f"\n  === Evaluation Phase ===")
            net.eval()
            eval_obs, eval_info = env.reset()
            eval_mcts = MCTSAgent(model=net, env=env, n_simulations=MCTS_SIMS*2, device=device, temperature=0.01)
            eval_steps = 0
            eval_done = False
            
            while not eval_done and eval_steps < config["max_rank"]:
                eval_action = eval_mcts.search(eval_obs)
                u, v, w = eval_mcts._parse_action(eval_action)
                eval_obs, eval_reward, eval_terminated, eval_truncated, eval_info = env.step({'u': u, 'v': v, 'w': w})
                eval_done = eval_terminated or eval_truncated
                eval_steps += 1
                
            print(f"  Eval: Steps={eval_steps}, Rank={eval_info['rank_used']}, Residual={eval_info['residual_norm']:.4e}, Success={eval_terminated}")
            
            # Track best solution
            if eval_terminated and eval_info['rank_used'] < best_rank_found:
                best_rank_found = eval_info['rank_used']
                print(f"  🎉 New best rank found: {best_rank_found}!")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            log_metrics({
                "eval_steps": eval_steps,
                "eval_rank": eval_info['rank_used'],
                "eval_residual": eval_info['residual_norm'],
                "eval_success": int(eval_terminated)
            }, step=epoch)
        
        # Save Checkpoint
        if (epoch+1) % 10 == 0:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_rank': best_rank_found,
                'replay_buffer': list(replay_buffer)[-1000:]  # Save last 1000 experiences
            }
            torch.save(checkpoint, f"checkpoints/model_epoch_{epoch+1}.pth")
            print(f"  Checkpoint saved!")
        
        # Early stopping
        if epochs_without_improvement >= 30:
            print(f"\n⚠️ No improvement for {epochs_without_improvement} eval cycles. Consider stopping or adjusting hyperparameters.")

if __name__ == "__main__":
    train()

