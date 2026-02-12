import sys
import os
import time
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
from project.logger import init_logger, log_metrics, finish_logger
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
    "run_name": "RLMM_denseNnet_and_coldStart",
    "matrix_size": (2, 2, 2),
    "max_rank": project_config.MAX_STEPS,  # Use config value (20 for full exploration)
    "learning rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "replay_buffer_size": REPLAY_BUFFER_SIZE,
    "epochs": EPOCHS,
    "episodes_per_epoch": EPISODES_PER_EPOCH,
    "mcts_simulations": 350,  # Production-ready simulation count
    "checkpoint_interval": 10,
    "device": "cpu"
}

# Global history for tracking benchmark progress across evaluations
BENCHMARK_HISTORY = []


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


def compute_marginal_targets(visit_counts, n_heads=12, action_map=[-1, 0, 1]):
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


def run_benchmark_evaluation(env, algorithm_data, eval_step_counter, global_step, is_final=False):
    """
    Run benchmark evaluation comparing naive, Strassen, and agent-discovered algorithms.
    Measures latency, multiplications, and correctness — logs everything to WandB.
    
    This runs on CPU using NumPy so it works regardless of GPU/CuPy availability.
    """
    print(f"\n  📊 Running {'Final ' if is_final else '' }Benchmark Evaluation...")
    
    # --- 1. Create test matrices ---
    np.random.seed(42)  # Reproducible benchmark
    A = np.random.rand(2, 2).astype(np.float32)
    B = np.random.rand(2, 2).astype(np.float32)
    C_ref = A @ B  # Reference result (NumPy matmul)
    
    n_warmup = 100
    n_iterations = 1000
    
    # --- 2. Naive algorithm (8 multiplications) ---
    def naive_matmul(A, B):
        C = np.zeros((2, 2), dtype=np.float32)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    C[i, j] += A[i, k] * B[k, j]
        return C
    
    # Warmup
    for _ in range(n_warmup):
        naive_matmul(A, B)
    
    # Timed runs
    naive_times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter_ns()
        C_naive = naive_matmul(A, B)
        t1 = time.perf_counter_ns()
        naive_times.append((t1 - t0) / 1000.0)  # Convert ns to us
    
    naive_latency = float(np.median(naive_times))
    naive_correct = float(np.max(np.abs(C_naive - C_ref))) < 1e-5
    naive_error = float(np.max(np.abs(C_naive - C_ref)))
    
    # --- 3. Strassen's algorithm (7 multiplications) ---
    def strassen_matmul(A, B):
        a00, a01, a10, a11 = A[0,0], A[0,1], A[1,0], A[1,1]
        b00, b01, b10, b11 = B[0,0], B[0,1], B[1,0], B[1,1]
        
        m1 = (a00 + a11) * (b00 + b11)
        m2 = (a10 + a11) * b00
        m3 = a00 * (b01 - b11)
        m4 = a11 * (b10 - b00)
        m5 = (a00 + a01) * b11
        m6 = (a10 - a00) * (b00 + b01)
        m7 = (a01 - a11) * (b10 + b11)
        
        C = np.zeros((2, 2), dtype=np.float32)
        C[0, 0] = m1 + m4 - m5 + m7
        C[0, 1] = m3 + m5
        C[1, 0] = m2 + m4
        C[1, 1] = m1 - m2 + m3 + m6
        return C
    
    for _ in range(n_warmup):
        strassen_matmul(A, B)
    
    strassen_times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter_ns()
        C_strassen = strassen_matmul(A, B)
        t1 = time.perf_counter_ns()
        strassen_times.append((t1 - t0) / 1000.0)
    
    strassen_latency = float(np.median(strassen_times))
    strassen_correct = float(np.max(np.abs(C_strassen - C_ref))) < 1e-5
    strassen_error = float(np.max(np.abs(C_strassen - C_ref)))
    
    # --- 4. Agent's algorithm ---
    rank1_tensors = algorithm_data.get('rank1_tensors', [])
    num_agent_mults = len(rank1_tensors)
    
    def agent_matmul(A, B, tensors):
        """Execute the agent's discovered decomposition."""
        A_flat = A.flatten()  # [a00, a01, a10, a11]
        B_flat = B.flatten()  # [b00, b01, b10, b11]
        C = np.zeros((2, 2), dtype=np.float32)
        
        for tensor in tensors:
            u = np.array(tensor['u'], dtype=np.float32)
            v = np.array(tensor['v'], dtype=np.float32)
            w = np.array(tensor['w'], dtype=np.float32)
            
            # Linear combo of A entries weighted by u
            a_combo = np.dot(u, A_flat)
            # Linear combo of B entries weighted by v
            b_combo = np.dot(v, B_flat)
            # Scalar multiplication
            product = a_combo * b_combo
            # Distribute to output via w
            C_flat = w * product
            C += C_flat.reshape(2, 2)
        
        return C
    
    for _ in range(n_warmup):
        agent_matmul(A, B, rank1_tensors)
    
    agent_times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter_ns()
        C_agent = agent_matmul(A, B, rank1_tensors)
        t1 = time.perf_counter_ns()
        agent_times.append((t1 - t0) / 1000.0)
    
    agent_latency = float(np.median(agent_times))
    agent_error = float(np.max(np.abs(C_agent - C_ref)))
    agent_correct = agent_error < 1e-5
    
    # --- 5. Compute derived metrics ---
    speedup_vs_naive = naive_latency / agent_latency if agent_latency > 0 else 0
    speedup_vs_strassen = strassen_latency / agent_latency if agent_latency > 0 else 0
    
    global BENCHMARK_HISTORY
    
    # --- 6. Initialize History with Baselines (if first run) ---
    if not BENCHMARK_HISTORY:
        # Columns: [Algorithm, Latency (us), Mults, Speedup, Correct]
        strassen_vs_naive = naive_latency / strassen_latency if strassen_latency > 0 else 0
        BENCHMARK_HISTORY.append(["Naive", round(naive_latency, 3), 8, "1.00x", "✅" if naive_correct else "❌"])
        BENCHMARK_HISTORY.append(["Strassen", round(strassen_latency, 3), 7, f"{strassen_vs_naive:.2f}x", "✅" if strassen_correct else "❌"])
    
    # --- 7. Append current Agent performance ---
    agent_label = f"Agent (Epoch {eval_step_counter*5})" if not is_final else "Final Agent"
    BENCHMARK_HISTORY.append([
        agent_label, 
        round(agent_latency, 3), 
        num_agent_mults, 
        f"{speedup_vs_naive:.2f}x", 
        "✅" if agent_correct else "❌"
    ])
    
    # --- 8. Log Standard WandB Table (Clean & Expandable) ---
    table = wandb.Table(
        columns=["Algorithm", "Latency (us)", "Multiplications", "Speedup (vs Naive)", "Correct"],
        data=BENCHMARK_HISTORY
    )
    
    # Logging the table. In WandB, click the 'Expand' icon on this tile to view full-screen & zoom.
    log_metrics({
        "Benchmark_Performance_Summary": table
    }, step=global_step)
    
    # --- 9. Log Bar Charts (Latest Comparison only) ---
    # Latency bar chart
    latency_table = wandb.Table(
        data=[
            ["Naive", naive_latency],
            ["Strassen", strassen_latency],
            ["Agent", agent_latency],
        ],
        columns=["Algorithm", "Latency (us)"]
    )
    log_metrics({
        "benchmark/latency_chart": wandb.plot.bar(
            latency_table, "Algorithm", "Latency (us)",
            title=f"Benchmark: Latency Comparison (us) {'- FINAL' if is_final else ''}"
        )
    }, step=global_step)
    
    # Multiplications bar chart
    mult_table = wandb.Table(
        data=[
            ["Naive", 8],
            ["Strassen", 7],
            ["Agent", num_agent_mults],
        ],
        columns=["Algorithm", "Multiplications"]
    )
    log_metrics({
        "benchmark/multiplications_chart": wandb.plot.bar(
            mult_table, "Algorithm", "Multiplications",
            title=f"Benchmark: Multiplication Count {'- FINAL' if is_final else ''}"
        )
    }, step=global_step)

    # Also update run summary for the overview page
    if wandb.run is not None:
        wandb.run.summary["Final_Benchmark_Summary"] = table
    
    # --- 9. Print summary (Console) ---
    print(f"  ┌────────────────────────────────────────────────────────────┐")
    print(f"  │  BENCHMARK RESULTS {'(FINAL)' if is_final else ''}                  │")
    print(f"  ├──────────────┬──────────────┬───────┬─────────────────────┤")
    print(f"  │ Algorithm    │ Latency (us) │ Mults │ Correct             │")
    print(f"  ├──────────────┼──────────────┼───────┼─────────────────────┤")
    print(f"  │ Naive        │ {naive_latency:>12.3f} │     8 │ {'✅' if naive_correct else '❌'}               │")
    print(f"  │ Strassen     │ {strassen_latency:>12.3f} │     7 │ {'✅' if strassen_correct else '❌'}               │")
    print(f"  │ Agent        │ {agent_latency:>12.3f} │ {num_agent_mults:>5} │ {'✅' if agent_correct else '❌'}               │")
    print(f"  ├──────────────┴──────────────┴───────┴─────────────────────┤")
    print(f"  │ Agent speedup vs Naive: {speedup_vs_naive:.2f}x                         │")
    print(f"  │ Agent speedup vs Strassen: {speedup_vs_strassen:.2f}x                      │")
    print(f"  └────────────────────────────────────────────────────────────┘")
    
    return {} # Return empty instead of scalars to avoid accidental logging 



def train():
    # 1. setup
    device = torch.device(config["device"])
    print(f"Training on device: {device}")
    
    # GPU info - only show if using CUDA
    if config["device"] == "cuda" and torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        torch.cuda.empty_cache()  # Clear cache before training
    elif config["device"] == "cuda":
        print("⚠️ WARNING: CUDA requested but not available. Falling back to CPU.")
    else:
        print("Using CPU for training as requested.")

    # initialize Environment, Network and Optimizer
    env = TensorDecompositionEnv(matrix_size=config["matrix_size"], max_rank=config["max_rank"])

    net = PolicyValueNet(
        input_dim=project_config.INPUT_DIM, 
        hidden_dim=project_config.HIDDEN_DIM, 
        n_heads=project_config.N_HEADS
    ).to(device)
    checkpoint_path = "checkpoints/latest_model.pth"

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
        
            # Calculate exploration parameters - maintain high exploration longer
            # For full exploration (up to 20 ranks), we need sustained exploration
            # progress = epoch / EPOCHS
            
            # Temperature: high early (1.5) -> moderate late (0.5)
            # temperature = max(0.5, 1.5 - progress * 1.0)
            
            # Epsilon for random action: stays high longer for more exploration
            # High early (30%), moderate mid (15%), low late (5%)
            # if progress < 0.3:
            #     eps_greedy = 0.30  # First 30% of training: heavy exploration
            # elif progress < 0.7:
            #     eps_greedy = 0.20  # Middle 40%: moderate exploration  
            # else:
            #     eps_greedy = max(0.05, 0.15 - (progress - 0.7) * 0.33)  # Final 30%: exploitation
        
            # # If stuck at 8 steps for many epochs, boost exploration significantly
            # if best_rank_found >= 8 and epoch > 30:
            #     eps_greedy = min(0.40, eps_greedy + 0.15)
            #     temperature = min(2.0, temperature + 0.5)
            #     print(f"⚠️ Still at {best_rank_found} steps after {epoch} epochs - BOOSTING exploration!")
            # elif best_rank_found == 7 and epoch > 50:
            #     # Found Strassen! Now try to find even better (6 step?)
            #     eps_greedy = min(0.35, eps_greedy + 0.10)
            #     print(f"✅ Found 7-step! Trying to find 6-step solution...")
        
            # print(f"Temperature: {temperature:.3f}, Epsilon-greedy: {eps_greedy:.3f}, Best rank: {best_rank_found}")
            # print(f"Max allowed steps: {config['max_rank']}")

            # --- A. Self-Play Phase (Data Collection) ---
            net.eval() # Set to evaluation mode for inference
        
            epoch_successes = 0  # Reset for this epoch

            for episode in range(EPISODES_PER_EPOCH):
                obs, info = env.reset()
                # Create fresh MCTS for each episode with current exploration params
                mcts = MCTSAgent(model=net, env=env, n_simulations=config["mcts_simulations"], 
                                device=device)

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
                    }, step=global_step)
                
                    # Calculate live reward/status for visualization
                    curr_norm = info["residual_norm"]
                    sqrt8 = 8 ** 0.5
                    
                    if curr_norm <= sqrt8:
                        current_val = 1 - curr_norm / sqrt8
                        decision_status = "SOLVED"
                    elif curr_norm <= 6:
                        current_val = (sqrt8 - curr_norm) / (6 - sqrt8)
                        decision_status = "PARTIAL"
                    else:
                        current_val = -1
                        decision_status = "SEARCHING"

                    # Broadcast to Visualizer
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

                # --------------------
                # Episode summary table
                # --------------------
                # import wandb
                # table = wandb.Table(columns=["Episode", "Reward", "Residual", "Rank"])
                # table.add_data(
                #     episode,
                #     reward,
                #     info["residual_norm"],
                #     info["rank_used"]
                # )
                # episode_step += 1
                # log_metrics({"episode_summary": table}, step=episode_step)

                # assign value (winner) to all steps in this episode
                # if solved, value =1 else -1

                res_norm = np.linalg.norm(obs)

                final_value = reward_func(terminated)

                # Track episode success
                episode_solved = (res_norm <= sqrt8)
                recent_episodes.append(1 if episode_solved else 0)
                recent_success_rate = sum(recent_episodes) / len(recent_episodes)
                
                if episode_solved:
                    epoch_successes += 1
                    total_successes += 1
                    print(f"  ✅ Episode {episode+1} SOLVED: Steps={steps}, Rank={info['rank_used']}, Residual={res_norm:.4e}")
                else:
                    print(f"  ❌ Episode {episode+1} FAILED: Steps={steps}, Result={final_value:.3f}, Residual={res_norm:.4e}")

                # --------------------
                # Episode Log
                # --------------------
                episode_step += 1
                log_metrics({
                    "episode/final_value": final_value, 
                    "episode/rank_used": info["rank_used"],
                    "episode/residual_norm": info["residual_norm"],
                    "episode_step": episode_step
                }, step=global_step)

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
                values_tensor = torch.FloatTensor(np.array(values)).unsqueeze(1).to(device) # shape (BATCH_SIZE, 1)

                # targets_batch is shape(BATCH_SIZE, n_heads, 3)
                targets_batch = torch.stack(list(target_dists)).to(device)

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
                # action_targets = torch.LongTensor(actions).to(device) # LongTensor for integer targets
                # # map {-1, 0, 1} to {0, 1, 2} for cross-entropy
                # action_targets = action_targets + 1

                # policy loss with soft targets (KL Divergence)
                # we treat each head independently
                policy_loss = 0

                for i in range(len(policy_logits)):
                    # network output: Logits -> LogSoftmax for KLDiv, shape: (batch_size, 3)
                    pred_log_probs = F.log_softmax(policy_logits[i], dim=1)

                    # target : probability distribution
                    # shape : (batch_size, 3)
                    target_probs = targets_batch[:, i, :]

                    # KLDivLoss(input = log_probs, target = probs)
                    # "batchmean" sums over classes and averages over batch
                    policy_loss += F.kl_div(pred_log_probs, target_probs, reduction='batchmean')
            
                # combine losses with weighted policy loss (policy is harder)
                loss = value_loss + 0.5 * policy_loss

                optimizer.zero_grad() # clear previous gradients
                loss.backward()       # backpropagate to compute gradients
            
                # Gradient clipping for stability and track gradient norms
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            
                optimizer.step()      # update weights
            
                # Track if gradients are vanishing/exploding
                if train_step == 0:  # Only for first batch per epoch
                    total_grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

                total_loss += loss.item() # accumulate loss for logging
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
                # "temperature": temperature,
                # "epsilon": eps_greedy
            })

            epoch_step += 1
            # Log epoch metrics
            log_metrics({
                "epoch_step" : epoch_step,
                "epoch/avg_loss": avg_loss,
                "epoch/value_loss": avg_value_loss,
                "epoch/policy_loss": avg_policy_loss,
                "epoch/learning_rate": current_lr,
                # "epoch/replay_buffer_size": len(replay_buffer),
                "epoch/best_rank_found": best_rank_found,
                # # "temperature": temperature,
                # # "epsilon": eps_greedy,
                # "epoch/epoch_successes": epoch_successes,
                # "epoch/total_successes": total_successes,
                "epoch/success_rate": recent_success_rate,
            }, step=global_step)
        
            # Step learning rate scheduler
            scheduler.step()

            # Evaluation phase (no exploration, no noise)
            if (epoch+1) % 5 == 0:  # More frequent evaluation
                print(f"\n  === Evaluation Phase ===")
                net.eval()
                eval_obs, eval_info = env.reset()
                eval_mcts = MCTSAgent(model=net, env=env, n_simulations=config["mcts_simulations"]*2, device=device, temperature=0.01)
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
                
                # Evaluation step counter for internal tracking
                eval_step += 1
                
                # ==========================================
                # BENCHMARK: Latency, Multiplications, etc.
                # ==========================================
                algorithm_data = env.get_algorithm_description()
                try:
                    run_benchmark_evaluation(env, algorithm_data, eval_step, global_step)
                except Exception as e:
                    print(f"  ⚠️ Benchmark evaluation failed: {e}")
        
            # Save Checkpoint
            if (epoch+1) % 10 == 0:
                if not os.path.exists("checkpoints"):
                    os.makedirs("checkpoints")
                torch.save(net.state_dict(), f"checkpoints/latest_model.pth")
    finally:
        print("Stopping Visualizer Server...")
        viz.stop()
    
    # ==========================================
    # FINAL BENCHMARK after all training
    # ==========================================
    try:
        print("\n" + "="*60)
        print("FINAL BENCHMARK - Post-Training Evaluation")
        print("="*60)
        
        # Run one last eval episode to get the final algorithm
        net.eval()
        final_obs, final_info = env.reset()
        final_mcts = MCTSAgent(model=net, env=env, n_simulations=config["mcts_simulations"]*2, device=device, temperature=0.01)
        final_done = False
        final_steps = 0
        
        while not final_done and final_steps < config["max_rank"]:
            final_action = final_mcts.search(final_obs, add_noise=False)
            u, v, w = final_mcts._parse_action(final_action)
            final_obs, _, final_terminated, final_truncated, final_info = env.step({'u': u, 'v': v, 'w': w})
            final_done = final_terminated or final_truncated
            final_steps += 1
        
        algorithm_data = env.get_algorithm_description()
        print(f"Final agent: {algorithm_data['num_multiplications']} multiplications, "
              f"residual={algorithm_data['verification']['residual_norm']:.4e}, "
              f"complete={algorithm_data['verification']['complete']}")
        
        eval_step += 1
        run_benchmark_evaluation(env, algorithm_data, eval_step, global_step, is_final=True)
        
        print("="*60)
    except Exception as e:
        print(f"Failed to run final benchmark: {e}")
    
    # Also run the GPU benchmark if CuPy is available (separate WandB run)
    try:
        project_root = Path(__file__).parent.parent
        print("\nAttempting GPU benchmark (separate run)...")
        # Use run with timeout to prevent hanging the whole script
        subprocess.run([sys.executable, str(project_root / "update_benchmark.py")], cwd=str(project_root), timeout=60)
        subprocess.run([sys.executable, str(project_root / "visualize_benchmark.py")], cwd=str(project_root), timeout=60)
    except subprocess.TimeoutExpired:
        print("⚠️ GPU benchmark timed out after 60s.")
    except Exception as e:
        print(f"GPU benchmark skipped: {e}")
    
    # Properly close WandB
    finish_logger()
    print("\n✅ Training complete. All metrics logged to WandB.")

if __name__ == "__main__":
    train()
