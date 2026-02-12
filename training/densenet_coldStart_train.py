import sys
import os
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import subprocess
import wandb

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.gym import TensorDecompositionEnv
from agent.mcts_agent import MCTSAgent
from models.linear_pv_network import PolicyValueNet
from project.logger import init_logger, log_metrics
import config as project_config
from utils.warm_start import generate_demo_data
from training.visualizer_server import VisualizerServer
from utils.reward import reward_func
# from mlo.checkpoint import save_checkpoint

# --- Hyperparameters (now from config) ---
LEARNING_RATE = project_config.LEARNING_RATE
BATCH_SIZE = project_config.BATCH_SIZE
REPLAY_BUFFER_SIZE = project_config.REPLAY_BUFFER_SIZE
EPOCHS = project_config.EPOCHS
EPISODES_PER_EPOCH = project_config.EPISODES_PER_EPOCH
MCTS_SIMS = project_config.MCTS_SIMS

config = {
    "run_name": "RLMM_denseNet_and_coldStart_3x3",
    "matrix_size": (3, 3, 3),  # Changed from (2, 2, 2) to (3, 3, 3)
    "max_rank": project_config.MAX_STEPS,  # Use config value (30 for full exploration)
    "learning rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "replay_buffer_size": REPLAY_BUFFER_SIZE,
    "epochs": EPOCHS,
    "episodes_per_epoch": EPISODES_PER_EPOCH,
    "mcts_simulations": MCTS_SIMS,
    "checkpoint_interval": 10,
    "device": "cpu"
}


class PrioritizedReplayBuffer:
    """
    Replay buffer that samples successful episodes more frequently.
    This is critical for learning from sparse rewards.
    """
    def __init__(self, maxlen, success_weight=3.0):
        self.buffer = deque(maxlen=maxlen)
        self.success_weight = success_weight
        
    def append(self, item):
        """item is (state, policy_target, value, is_success)"""
        self.buffer.append(item)
        
    def extend(self, items):
        for item in items:
            self.append(item)
    
    def sample(self, batch_size):
        """Sample with priority for successful episodes."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        # Calculate weights: successful samples get higher weight
        weights = []
        for item in self.buffer:
            is_success = item[3] if len(item) > 3 else (item[2] > 0.5)
            weight = self.success_weight if is_success else 1.0
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        # Sample with replacement according to weights
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=True, p=probs)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)


def compute_marginal_targets(visit_counts, n_heads=27, action_map=[-1, 0, 1]):  # Changed from 12 to 27
    """
    Converts MCTS tuple visits into marginal probability targets for each head.
    
    Args:
        visit_counts: Dict { (1, 0, -1...): 50, (0, 0, 1...): 20 }
    Returns:
        targets: Tensor of shape (n_heads, 3) containing probabilities.
    """
    # 1. calculate total visits to normalize
    total_visits = sum(visit_counts.values())
    if total_visits == 0:
        return torch.zeros(n_heads, 3)
    
    # 2. intialize targets(Heads x Actions)
    # actions are indices 0,1,2 corresponding to -1,0,1
    # shape (n_heads, 3)
    targets = torch.zeros(n_heads, len(action_map))

    # 3. aggregrate
    for action_tuple, count in visit_counts.items():
        prob = count / total_visits

        # for each scalar decision in the tuple
        for head_idx, action in enumerate(action_tuple):
            # map value (-1,0,1) to index (0,1,2)
            # assuming action_map is sorted as [-1,0,1]
            action_idx = action_map.index(action)

            # add probability mass to this choice
            targets[head_idx, action_idx] += prob # add to the corresponding head and action index

    return targets


def pretrain_on_expert(net, replay_buffer, device, optimizer, steps=50):
    """Pre-train the network on expert demonstrations."""
    print(f"\n{'='*50}")
    print("Pre-training on expert demonstrations...")
    print(f"{'='*50}")
    
    net.train()
    for step in range(steps):
        if len(replay_buffer) < BATCH_SIZE:
            continue
            
        batch = replay_buffer.sample(BATCH_SIZE)
        states, target_dists, values, _ = zip(*batch)

        states_tensor = torch.FloatTensor(np.array(states)).to(device)
        values_tensor = torch.FloatTensor(np.array(values)).unsqueeze(1).to(device)
        targets_batch = torch.stack(list(target_dists)).to(device)

        policy_logits, value_pred = net(states_tensor)

        # Value loss
        value_loss = F.mse_loss(value_pred, values_tensor)

        # Policy loss
        policy_loss = 0
        for i in range(len(policy_logits)):
            pred_log_probs = F.log_softmax(policy_logits[i], dim=1)
            target_probs = targets_batch[:, i, :]
            policy_loss += F.kl_div(pred_log_probs, target_probs, reduction='batchmean')

        loss = value_loss + policy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        
        if step % 10 == 0:
            print(f"  Pre-train step {step}/{steps}: Loss={loss.item():.4f} (V={value_loss.item():.4f}, P={policy_loss.item():.4f})")
    
    print("Pre-training complete!")
    print(f"{'='*50}\n")


def train():
    # 1. setup
    device = torch.device(config["device"])
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

    net = PolicyValueNet(
        input_dim=project_config.INPUT_DIM, 
        hidden_dim=project_config.HIDDEN_DIM, 
        n_heads=project_config.N_HEADS
    ).to(device)
    checkpoint_path = "checkpoints/dense_cold_3x3/latest_model.pth"  # Updated checkpoint path

    # Check if a saved model already exists
    # NOTE: With the new architecture, old checkpoints won't be compatible
    # You should delete/rename old checkpoints to start fresh with the new model
    if os.path.exists(checkpoint_path):
        print("Found existing checkpoint...")
        try:
            net.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("✅ Successfully loaded checkpoint!")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint (architecture mismatch?): {e}")
            print("Starting with fresh weights instead.")
    else:
        print("No checkpoint found. Starting from scratch.")

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-5)

    # Prioritized Replay Buffer: samples successful episodes more frequently
    replay_buffer = PrioritizedReplayBuffer(maxlen=REPLAY_BUFFER_SIZE, success_weight=5.0)
    
    # Track best performance
    best_rank_found = config["max_rank"]
    epochs_without_improvement = 0
    total_successes = 0
    recent_success_rate = 0.0
    recent_episodes = deque(maxlen=50)  # Track last 50 episodes for success rate

    # ============================================
    # WARM START WITH EXPERT DEMONSTRATIONS
    # This is CRITICAL for learning to work!
    # ============================================
    # print("\n" + "="*60)
    # print("INITIALIZING WARM START WITH EXPERT DEMONSTRATIONS")
    # print("="*60)
    
    # demo_data = generate_demo_data(env)
    # # Convert demo data format: (state, target, value) -> (state, target, value, is_success)
    # demo_with_success = [(s, t, v, True) for s, t, v in demo_data]
    
    # # Inject expert demonstrations multiple times
    # for _ in range(project_config.WARM_START_COPIES):
    #     replay_buffer.extend(demo_with_success)

    # print(f"✅ Replay Buffer initialized with {len(replay_buffer)} expert samples.")

    # # Pre-train the network on expert data before starting MCTS
    # pretrain_on_expert(net, replay_buffer, device, optimizer, steps=project_config.PRE_TRAIN_STEPS)

    # --------------------
    # 2. Main Training Loop
    # --------------------

    init_logger(config)

    global_step = 0
    epoch_step = 0
    eval_step = 0
    episode_step = 0
    
    # Start Visualizer Server
    viz = VisualizerServer()
    try:
        viz.start()
    except Exception as e:
        print(f"Failed to start Visualizer Server: {e}")

    try:
        for epoch in range(EPOCHS):
            print(f"\n{'='*60}")
            print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
            print(f"{'='*60}")

            # --- A. Self-Play Phase (Data Collection) ---
            net.eval() # Set to evaluation mode for inference
        
            epoch_successes = 0  # Reset for this epoch

            for episode in range(EPISODES_PER_EPOCH):
                obs, info = env.reset()
                # Create fresh MCTS for each episode with current exploration params
                mcts = MCTSAgent(model=net, env=env, n_simulations=MCTS_SIMS, 
                                device=config["device"])

                episode_data = [] # Stores (state, action_dist) for this game
                steps = 0
                episode_reward = 0
                valid_actions = 0
                invalid_actions = 0
                done = False
                final_value = 0
                terminated = False

                print(f"  Starting Episode {episode+1}...", flush=True)
                
                # play one full game
                while not done:
                    # run MCTS to get the best action distribution
                    # add_noise=True during training for exploration
                    best_action, visit_counts = mcts.search(obs, return_probs=True, add_noise=True)

                    # execute move (environment step)
                    u, v, w = mcts._parse_action(best_action)
                    action = {
                        'u': u,
                        'v': v,
                        'w': w
                    }
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    # compute visit marginals for training target
                    marginal_targets = compute_marginal_targets(visit_counts, n_heads=project_config.N_HEADS)

                    # store data : (state, targets, result_placeholder)
                    episode_data.append([obs.flatten(), marginal_targets])

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
                    "global_step": global_step,
                    "step/reward": reward,
                    "step/residual_norm": info["residual_norm"],
                    "step/rank_used": info["rank_used"],
                    "step/action_valid": int(info["action_valid"]),
                    "step/action_sparsity": action_sparsity,
                    "step/action": action
                    })
                
                    # Calculate live reward/status for visualization (updated for 3x3)
                    curr_norm = info["residual_norm"]
                    
                    viz.broadcast({
                        "type": "step",
                        "global_step": global_step,
                        "episode": episode,
                        "reward": current_val,
                        "residual": curr_norm,
                        "rank": info["rank_used"],
                        "step_count": steps,
                        "elapsed": 0,
                        "sparsity": action_sparsity,
                        "action_valid": int(info["action_valid"]),
                        "action": action,
                        "status": decision_status
                    })

                    print(
                    f"Episode {episode:03d} | "
                    f"Reward: {current_val:7.2f} | "
                    f"Rank: {info['rank_used']} | "
                    f"Residual: {info['residual_norm']:.4e} | "
                    f"Action: {action} | "
                    f"Steps: {steps} | "
                    f"Observation residual: {np.linalg.norm(obs):.2f} | "
                    )
                
                # end of one episode

                # assign value (winner) to all steps in this episode
                # if solved, value =1 else -1

                res_norm = np.linalg.norm(obs)

                final_value = reward_func(terminated)

                # Track episode success (updated threshold for 3x3)
                episode_solved = (res_norm < 1e-5)  # Consider solved if residual is effectively zero
                recent_episodes.append(1 if episode_solved else 0)
                recent_success_rate = sum(recent_episodes) / len(recent_episodes)
                
                if episode_solved:
                    epoch_successes += 1
                    total_successes += 1
                    print(f"  ✅ Episode {episode+1} SOLVED: Steps={steps}, Rank={info['rank_used']}, Residual={res_norm:.4e}")
                else:
                    print(f"  ❌ Episode {episode+1} FAILED: Steps={steps}, Result={final_value:.3f}, Residual={res_norm:.4e}")

                # --------------------
                # Episode summary table
                # --------------------
                episode_step += 1
                table = wandb.Table(columns=["Epoch", "Episode", "Reward", "Residual", "Rank", "Solved"])
                table.add_data(
                    epoch,
                    episode,
                    final_value,
                    info["residual_norm"],
                    info["rank_used"],
                    episode_solved
                )

                log_metrics({"episode/summary": table,"episode/final_value":final_value, "episode_step" : episode_step})

                # Store episode data with success flag for prioritized replay
                for data in episode_data:
                    state_flat, policy_target = data
                    replay_buffer.append((state_flat, policy_target, final_value, episode_solved))

            # --- B. Training Phase (Network Update)--- once per Epoch ( after several episodes )

            net.train() # set to training mode
            if len(replay_buffer) < BATCH_SIZE:
                print("Not enough data in replay buffer to train.")
                continue

            # run more updates on the buffer (increased from 10)
            total_loss = 0
            total_value_loss = 0
            total_policy_loss = 0
            total_grad_norm = 0.0
            n_train_steps = 20  # Increased from 10
        
            for train_step in range(n_train_steps):
                batch = replay_buffer.sample(BATCH_SIZE)
                states, target_dists, values, _ = zip(*batch)  # Unpack 4 elements now

                # prepare tensors
                states_tensor = torch.FloatTensor(np.array(states)).to(device)
                values_tensor = torch.FloatTensor(np.array(values)).unsqueeze(1).to(device)

                # targets_batch is shape(BATCH_SIZE, n_heads, 3)
                targets_batch = torch.stack(list(target_dists)).to(device)

                # forward pass
                policy_logits, value_pred = net(states_tensor)
            
                # 1. Value Loss (MSE)
                value_loss = F.mse_loss(value_pred, values_tensor)

                # 2. Policy Loss (KL Divergence)
                policy_loss = 0

                for i in range(len(policy_logits)):
                    pred_log_probs = F.log_softmax(policy_logits[i], dim=1)
                    target_probs = targets_batch[:, i, :]
                    policy_loss += F.kl_div(pred_log_probs, target_probs, reduction='batchmean')
            
                # combine losses with weighted policy loss
                loss = value_loss + 0.5 * policy_loss

                optimizer.zero_grad()
                loss.backward()
            
                # Gradient clipping for stability
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
                optimizer.step()
            
                # Track gradients
                if train_step == 0:
                    total_grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

                total_loss += loss.item()
                total_value_loss += value_loss.item()
                total_policy_loss += policy_loss.item()

            avg_loss = total_loss / n_train_steps
            avg_value_loss = total_value_loss / n_train_steps
            avg_policy_loss = total_policy_loss / n_train_steps

            current_lr = optimizer.param_groups[0]['lr']
        
            print(f"  Average Loss: {avg_loss:.4f} (Value: {avg_value_loss:.4f}, Policy: {avg_policy_loss:.4f})")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Gradient Norm: {total_grad_norm:.4f}")
        
            # Broadcast training metrics to visualizer
            viz.broadcast({
                "type": "training",
                "epoch": epoch,
                "loss": avg_loss,
                "policy_loss": avg_policy_loss,
                "value_loss": avg_value_loss,
                "learning_rate": current_lr,
                "gradient_norm": total_grad_norm,
                "replay_buffer_size": len(replay_buffer),
                "success_rate": recent_success_rate
            })
        
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
        
            # Broadcast epoch summary
            viz.broadcast({
                "type": "epoch_summary",
                "epoch": epoch,
                "epoch_successes": epoch_successes,
                "total_successes": total_successes,
                "success_rate": recent_success_rate,
                "best_rank": best_rank_found,
            })

            epoch_step += 1
            # Log epoch metrics
            log_metrics({
                "epoch_step" : epoch_step,
                "epoch/avg_loss": avg_loss,
                "epoch/value_loss": avg_value_loss,
                "epoch/policy_loss": avg_policy_loss,
                "epoch/learning_rate": current_lr,
                "epoch/best_rank_found": best_rank_found,
                "epoch/success_rate": recent_success_rate,
            })
        
            # Step learning rate scheduler
            scheduler.step()

            # Evaluation phase (no exploration, no noise)
            if (epoch+1) % 5 == 0:  # More frequent evaluation
                print(f"\n  === Evaluation Phase ===")
                net.eval()
                eval_obs, eval_info = env.reset()
                eval_mcts = MCTSAgent(model=net, env=env, n_simulations=MCTS_SIMS*2, device=device, temperature=0.01)
                eval_steps = 0
                eval_done = False
            
                while not eval_done and eval_steps < config["max_rank"]:
                    # No noise during evaluation for deterministic results
                    eval_action = eval_mcts.search(eval_obs, add_noise=False)
                    u, v, w = eval_mcts._parse_action(eval_action)
                    eval_obs, eval_reward, eval_terminated, eval_truncated, eval_info = env.step({'u': u, 'v': v, 'w': w})
                    eval_done = eval_terminated or eval_truncated
                    eval_steps += 1
                
                eval_residual = np.linalg.norm(eval_obs[:project_config.TENSOR_DIM])
                print(f"  Eval: Steps={eval_steps}, Rank={eval_info['rank_used']}, Residual={eval_residual:.4e}, Success={eval_terminated}")
            
                # Track best solution
                if eval_terminated and eval_info['rank_used'] < best_rank_found:
                    best_rank_found = eval_info['rank_used']
                    print(f"  🎉 New best rank found: {best_rank_found}!")
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                
                eval_step += 1
                log_metrics({
                    "eval_step":eval_step,
                    "eval/steps": eval_steps,
                    "eval/rank": eval_info['rank_used'],
                    "eval/residual": eval_info['residual_norm'],
                    "eval/success": int(eval_terminated)
                })
        
            # Save Checkpoint
            if (epoch+1) % 10 == 0:
                if not os.path.exists("checkpoints/dense_cold_3x3"):  # Updated path
                    os.makedirs("checkpoints/dense_cold_3x3")
                torch.save(net.state_dict(), f"checkpoints/dense_cold_3x3/latest_model.pth")
                print(f"Checkpoint saved!")
    finally:
        print("Stopping Visualizer Server...")
        viz.stop()
    
    # Run benchmark visualization after training
    try:
        print("\n" + "="*50)
        print("Running benchmark update with trained agent...")
        subprocess.call([sys.executable, "update_benchmark.py"])
        subprocess.call([sys.executable, "visualize_benchmark.py"])
        print("="*50)
    except Exception as e:
        print(f"Failed to run benchmark update: {e}")

if __name__ == "__main__":
    train()

