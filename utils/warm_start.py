import numpy as np
import torch
import config

# helper to ensure data consistency with MCTS
def convert_tuple_to_target(action_tuple, n_heads=config.N_HEADS):
    """
    Converts a hard integer tuple (e.g. 1, -1, 0) into a 
    one-hot tensor probability distribution (shape 12x3 or 27x3).
    """
    # Map: -1 -> 0, 0 -> 1, 1 -> 2
    action_map = {-1: 0, 0: 1, 1: 2}

    # Initialize tensor of zeros
    target = torch.zeros(n_heads, 3)

    for i, val in enumerate(action_tuple):
        idx = action_map[val]
        target[i, idx] = 1.0  # Set probability to 100% for the expert move
        
    return target

def get_standard_2x2_demo():
    """
    Returns a list of (state, action_tuple, reward) for the 
    Standard (Naive) Matrix Multiplication algorithm (8 steps).
    """
    # 2x2 Matrix Multiplication (C = A * B)
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22

    # We define the 8 moves (Terms) explicitly.
    # Format: (u indices, v indices, w indices) 
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1) for 2x2
    
    # u=[1,0,0,0] -> A11, etc.

    moves = []

    # 1. A11 * B11 -> C11
    moves.append(([1,0,0,0], [1,0,0,0], [1,0,0,0]))
    # 2. A12 * B21 -> C11
    moves.append(([0,1,0,0], [0,0,1,0], [1,0,0,0]))
    
    # 3. A11 * B12 -> C12
    moves.append(([1,0,0,0], [0,1,0,0], [0,1,0,0]))
    # 4. A12 * B22 -> C12
    moves.append(([0,1,0,0], [0,0,0,1], [0,1,0,0]))
    
    # 5. A21 * B11 -> C21
    moves.append(([0,0,1,0], [1,0,0,0], [0,0,1,0]))
    # 6. A22 * B21 -> C21
    moves.append(([0,0,0,1], [0,0,1,0], [0,0,1,0]))
    
    # 7. A21 * B12 -> C22
    moves.append(([0,0,1,0], [0,1,0,0], [0,0,0,1]))
    # 8. A22 * B22 -> C22
    moves.append(([0,0,0,1], [0,0,0,1], [0,0,0,1]))
    
    return moves

def generate_demo_data(env):
    """
    Plays the standard algorithm on the environment and captures the
    States and Actions to fill the Replay Buffer.
    """
    demo_buffer = []
    moves = get_standard_2x2_demo()
    state, info = env.reset()

    print("Generating Warm-Start Data (Standard Algo)...")

    for u_list, v_list, w_list in moves:
        # convert list to numpy arrays
        u_np = np.array(u_list)
        v_np = np.array(v_list)
        w_np = np.array(w_list)

        # convert list to Flat tuple for Agent Training Target
        # u + v + w
        action_tuple = tuple(u_list + v_list + w_list)

        # convert tuple to Tensor Target
        soft_target = convert_tuple_to_target(action_tuple, n_heads=config.N_HEADS)

        # store current state and the "correct" action
        # result is 1.0 because we know this leads to a solution
        demo_buffer.append((state.flatten(), soft_target, 1.0))

        action = {
                    'u': u_np,
                    'v': v_np,
                    'w': w_np
                }

        # step env to get next state
        next_obs, reward, terminated, truncated, info = env.step(action)

    if np.linalg.norm(next_obs) < 1e-5:
        print("Verification: Standard Algorithm successfully zeroed the tensor!")
    else:
        print("Warning: Standard Algorithm did not fully solve the tensor. Check indices.")
    
    return demo_buffer