import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import os
from collections import deque

# Import our project modules
import config
from env_tmp.matrix_env import MatrixTensorEnv
from models.pv_network import PolicyValueNet
from agent.mcts_agent import MCTSAgent

# --- Hyperparameters ---
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 5000
EPOCHS = 100         # Total training loops
EPISODES_PER_EPOCH = 5  # Self-play games per loop
MCTS_SIMS = 50       # Search depth per move

def train():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Initialize Environment, Network, and Optimizer
    env = MatrixTensorEnv(n=config.MATRIX_SIZE)
    net = PolicyValueNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    
    # Replay Buffer: Stores (state, mcts_probs, winner)
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    # 2. Main Training Loop
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # --- A. Self-Play Phase (Data Collection) ---
        net.eval() # Set to evaluation mode for inference
        
        for episode in range(EPISODES_PER_EPOCH):
            state = env.reset()
            mcts = MCTSAgent(net, env, n_simulations=MCTS_SIMS, device=device)
            
            episode_data = [] # Stores (state, action_dist) for this game
            steps = 0
            done = False
            
            # Play one full game
            while not done:
                # Run MCTS to get the best action distribution
                # Note: We need MCTS to return probabilities, not just the best move
                # We'll peek into the root node of the MCTS after search
                best_action = mcts.search(state)
                
                # Extract Visit Counts from Root -> Policy Target
                # We need to map visit counts to the 27 output heads.
                # This is complex. For simplicity in Phase 1, we train on the CHOSEN action.
                # Ideally, we would train on the full distribution vector.
                
                # Execute Move
                u, v, w = mcts._parse_action(best_action)
                next_state, reward, done, _ = env.step(u, v, w)
                
                # Store Data: (State, Action_Tuple, Result_Placeholder)
                episode_data.append([state.flatten(), best_action])
                
                state = next_state
                steps += 1
                if steps >= config.MAX_STEPS:
                    done = True

            # Assign Value (Winner) to all steps in this episode
            # If solved (reward > 0), value = 1. Else value = -1.
            final_value = 1.0 if reward > 0 else -1.0
            print(f"  Episode {episode+1}: Steps={steps}, Result={final_value}")
            
            for data in episode_data:
                state_flat, action_tuple = data
                replay_buffer.append((state_flat, action_tuple, final_value))

        # --- B. Training Phase (Network Update) ---
        net.train()
        if len(replay_buffer) < BATCH_SIZE:
            continue
            
        # Run a few updates on the buffer
        total_loss = 0
        for _ in range(10): # 10 Gradient steps per epoch
            batch = random.sample(replay_buffer, BATCH_SIZE)
            states, actions, values = zip(*batch)
            
            # Prepare Tensors
            states_tensor = torch.FloatTensor(np.array(states)).to(device)
            values_tensor = torch.FloatTensor(np.array(values)).unsqueeze(1).to(device)
            
            # Forward Pass
            policy_logits, value_pred = net(states_tensor)
            
            # 1. Value Loss (MSE): Did we predict the win/loss correctly?
            value_loss = F.mse_loss(value_pred, values_tensor)
            
            # 2. Policy Loss (Cross Entropy): Did we pick the right integers?
            # We must split the 'actions' tuple (batch of 27 ints) into 27 separate targets
            # actions is list of tuples: [(1, 0, -1...), ...]
            # convert to tensor: (batch, 27)
            action_targets = torch.LongTensor(actions).to(device)
            # Map {-1, 0, 1} to indices {0, 1, 2} for CrossEntropy
            action_targets = action_targets + 1 
            
            policy_loss = 0
            # Iterate through all 27 heads
            for i in range(len(policy_logits)):
                # Head i output: (batch, 3)
                # Target i: (batch) -> The i-th integer of the action tuple
                head_loss = F.cross_entropy(policy_logits[i], action_targets[:, i])
                policy_loss += head_loss
            
            # Combine Losses
            loss = value_loss + policy_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"  Avg Loss: {total_loss/10:.4f}")
        
        # Save Checkpoint
        if (epoch+1) % 10 == 0:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            torch.save(net.state_dict(), f"checkpoints/model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()