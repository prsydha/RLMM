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

def get_standard_3x3_demo():
    """
    Returns a list of (state, action_tuple, reward) for the 
    Standard (Naive) Matrix Multiplication algorithm for 3x3 (27 steps).
    
    3x3 Matrix Multiplication (C = A * B)
    C_ij = sum_k(A_ik * B_kj) for i,j,k in {0,1,2}
    
    Total: 27 multiplication terms (3^3)
    """
    # Index mapping for 3x3: 0=(0,0), 1=(0,1), 2=(0,2), 3=(1,0), 4=(1,1), 5=(1,2), 6=(2,0), 7=(2,1), 8=(2,2)
    # Each vector u, v, w has 9 elements
    
    moves = []
    
    # Generate all 27 terms: C[i,j] += A[i,k] * B[k,j]
    for i in range(3):  # row of A and C
        for j in range(3):  # column of B and C
            for k in range(3):  # column of A, row of B
                # Create one-hot vectors
                u = [0] * 9  # A matrix selector
                v = [0] * 9  # B matrix selector
                w = [0] * 9  # C matrix selector
                
                # A[i,k] position
                a_idx = i * 3 + k
                u[a_idx] = 1
                
                # B[k,j] position
                b_idx = k * 3 + j
                v[b_idx] = 1
                
                # C[i,j] position
                c_idx = i * 3 + j
                w[c_idx] = 1
                
                moves.append((u, v, w))
    
    return moves


def get_laderman_3x3_demo():
    """
    Returns the complete Laderman's algorithm for 3x3 matrix multiplication (23 operations).
    This is the proven optimal algorithm for 3x3 discovered by Julian D. Laderman in 1976.
    
    Each tensor uses coefficients from {-1, 0, 1} to form linear combinations.
    The 23 rank-1 tensors u ⊗ v ⊗ w completely determine the matrix product C = A × B.
    
    Index mapping for 3x3 matrices:
    0=(0,0), 1=(0,1), 2=(0,2)
    3=(1,0), 4=(1,1), 5=(1,2)
    6=(2,0), 7=(2,1), 8=(2,2)
    """
    moves = []
    
    # Laderman's 23 rank-1 tensors with proper {-1, 0, 1} coefficients
    
    # T1: (a00 + a01 + a02 - a10 - a11 - a21) ⊗ (b00) ⊗ (c01 + c10)
    moves.append(([1,1,1,-1,-1,0,-1,0,0], [1,0,0,0,0,0,0,0,0], [0,1,0,1,0,0,0,0,0]))
    
    # T2: (a00 - a10) ⊗ (b01 + b11) ⊗ (c00 + c10 + c11)
    moves.append(([1,0,0,-1,0,0,0,0,0], [0,1,0,0,1,0,0,0,0], [1,0,0,1,1,0,0,0,0]))
    
    # T3: (a11 + a21) ⊗ (b00 - b01 + b10 - b11 - b12 - b20) ⊗ (c01)
    moves.append(([0,0,0,0,1,0,0,1,0], [1,-1,0,1,-1,-1,-1,0,0], [0,1,0,0,0,0,0,0,0]))
    
    # T4: (a00 - a20) ⊗ (b02 + b12) ⊗ (c00 + c20 + c22)
    moves.append(([1,0,0,0,0,0,-1,0,0], [0,0,1,0,0,1,0,0,0], [1,0,0,0,0,0,1,0,1]))
    
    # T5: (a11 + a12 + a20 + a21 + a22) ⊗ (b11) ⊗ (c02 + c20)
    moves.append(([0,0,0,0,1,1,1,1,1], [0,0,0,0,1,0,0,0,0], [0,0,1,0,0,0,1,0,0]))
    
    # T6: (a22) ⊗ (b00 + b01 + b02 - b10 - b11 - b21) ⊗ (c02 + c20)
    moves.append(([0,0,0,0,0,0,0,0,1], [1,1,1,-1,-1,0,0,-1,0], [0,0,1,0,0,0,1,0,0]))
    
    # T7: (a01 - a11) ⊗ (b10 + b11) ⊗ (c00 + c01 + c11)
    moves.append(([0,1,0,0,-1,0,0,0,0], [0,0,0,1,1,0,0,0,0], [1,1,0,0,1,0,0,0,0]))
    
    # T8: (a12 + a22) ⊗ (b10 - b11 + b20 - b21 - b22) ⊗ (c02)
    moves.append(([0,0,0,0,0,1,0,0,1], [0,0,0,1,-1,0,1,-1,-1], [0,0,1,0,0,0,0,0,0]))
    
    # T9: (a02 - a22) ⊗ (b20 + b21) ⊗ (c00 + c02 + c22)
    moves.append(([0,0,1,0,0,0,0,0,-1], [0,0,0,0,0,0,1,1,0], [1,0,1,0,0,0,0,0,1]))
    
    # T10: (a00 + a01 + a02 - a11 - a12) ⊗ (b12) ⊗ (c01 + c11)
    moves.append(([1,1,1,0,-1,-1,0,0,0], [0,0,0,0,0,1,0,0,0], [0,1,0,0,1,0,0,0,0]))
    
    # T11: (a10 + a11 + a12) ⊗ (b02 - b12) ⊗ (c10 + c11 + c12)
    moves.append(([0,0,0,1,1,1,0,0,0], [0,0,1,0,0,-1,0,0,0], [0,0,0,1,1,1,0,0,0]))
    
    # T12: (a10 + a11 + a12 - a20 - a21) ⊗ (b22) ⊗ (c12 + c21)
    moves.append(([0,0,0,1,1,1,-1,-1,0], [0,0,0,0,0,0,0,0,1], [0,0,0,0,0,1,0,1,0]))
    
    # T13: (a00) ⊗ (b00 + b01 + b02) ⊗ (c00)
    moves.append(([1,0,0,0,0,0,0,0,0], [1,1,1,0,0,0,0,0,0], [1,0,0,0,0,0,0,0,0]))
    
    # T14: (a01 + a02) ⊗ (b10 - b20) ⊗ (c00 + c02)
    moves.append(([0,1,1,0,0,0,0,0,0], [0,0,0,1,0,0,-1,0,0], [1,0,1,0,0,0,0,0,0]))
    
    # T15: (a11 + a12) ⊗ (b10 + b11 + b12) ⊗ (c11)
    moves.append(([0,0,0,0,1,1,0,0,0], [0,0,0,1,1,1,0,0,0], [0,0,0,0,1,0,0,0,0]))
    
    # T16: (a20 + a21 + a22) ⊗ (b02 - b12) ⊗ (c20 + c21 + c22)
    moves.append(([0,0,0,0,0,0,1,1,1], [0,0,1,0,0,-1,0,0,0], [0,0,0,0,0,0,1,1,1]))
    
    # T17: (a10) ⊗ (b00 + b01 + b02) ⊗ (c10)
    moves.append(([0,0,0,1,0,0,0,0,0], [1,1,1,0,0,0,0,0,0], [0,0,0,1,0,0,0,0,0]))
    
    # T18: (a20) ⊗ (b00 + b01 + b02) ⊗ (c20)
    moves.append(([0,0,0,0,0,0,1,0,0], [1,1,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0]))
    
    # T19: (a00 + a10 + a20) ⊗ (b00) ⊗ (c00 + c10 + c20)
    moves.append(([1,0,0,1,0,0,1,0,0], [1,0,0,0,0,0,0,0,0], [1,0,0,1,0,0,1,0,0]))
    
    # T20: (a01 + a11 + a21) ⊗ (b01) ⊗ (c01 + c11 + c21)
    moves.append(([0,1,0,0,1,0,0,1,0], [0,1,0,0,0,0,0,0,0], [0,1,0,0,1,0,0,1,0]))
    
    # T21: (a02 + a12 + a22) ⊗ (b02) ⊗ (c02 + c12 + c22)
    moves.append(([0,0,1,0,0,1,0,0,1], [0,0,1,0,0,0,0,0,0], [0,0,1,0,0,1,0,0,1]))
    
    # T22: (a00 + a10 + a20) ⊗ (b10 + b20) ⊗ (c10 + c20)
    moves.append(([1,0,0,1,0,0,1,0,0], [0,0,0,1,0,0,1,0,0], [0,0,0,1,0,0,1,0,0]))
    
    # T23: (a01 + a11 + a21) ⊗ (b11 + b21) ⊗ (c11 + c21)
    moves.append(([0,1,0,0,1,0,0,1,0], [0,0,0,0,1,0,0,1,0], [0,0,0,0,1,0,0,1,0]))
    
    return moves


def generate_demo_data(env, include_laderman=False):
    """
    Plays the standard algorithm on the environment and captures the
    States and Actions to fill the Replay Buffer.
    
    Also optionally includes Laderman's algorithm for variety (if implemented).
    """
    demo_buffer = []
    
    # Standard 27-step algorithm for 3x3
    moves = get_standard_3x3_demo()
    state, info = env.reset()

    print("Generating Warm-Start Data (Standard 3x3 Algo - 27 steps)...")

    for u_list, v_list, w_list in moves:
        # convert list to numpy arrays
        u_np = np.array(u_list)
        v_np = np.array(v_list)
        w_np = np.array(w_list)

        # convert list to Flat tuple for Agent Training Target
        action_tuple = tuple(u_list + v_list + w_list)

        # convert tuple to Tensor Target
        soft_target = convert_tuple_to_target(action_tuple, n_heads=config.N_HEADS)

        # store current state and the "correct" action
        demo_buffer.append((state.flatten(), soft_target, 1.0))

        action = {
                    'u': u_np,
                    'v': v_np,
                    'w': w_np
                }

        # step env to get next state
        next_obs, reward, terminated, truncated, info = env.step(action)
        state = next_obs

    # Check residual using first 729 elements (tensor part for 3x3)
    residual_norm = np.linalg.norm(next_obs[:729])
    if residual_norm < 1e-5:
        print("✅ Standard 3x3 Algorithm successfully zeroed the tensor!")
    else:
        print(f"⚠️ Warning: Standard 3x3 Algorithm residual norm = {residual_norm:.4f}")
    
    # Optionally add Laderman's algorithm demos
    if include_laderman:
        print("\nGenerating Warm-Start Data (Laderman's 3x3 Algo - 23 steps)...")
        laderman_moves = get_laderman_3x3_demo()
        state, info = env.reset()
        
        for u_list, v_list, w_list in laderman_moves:
            u_np = np.array(u_list)
            v_np = np.array(v_list)
            w_np = np.array(w_list)
            
            action_tuple = tuple(u_list + v_list + w_list)
            soft_target = convert_tuple_to_target(action_tuple, n_heads=config.N_HEADS)
            demo_buffer.append((state.flatten(), soft_target, 1.0))
            
            action = {'u': u_np, 'v': v_np, 'w': w_np}
            next_obs, reward, terminated, truncated, info = env.step(action)
            state = next_obs
        
        residual_norm = np.linalg.norm(next_obs[:729])
        if residual_norm < 1e-5:
            print("✅ Laderman's 3x3 Algorithm successfully zeroed the tensor!")
        else:
            print(f"⚠️ Warning: Laderman's Algorithm residual norm = {residual_norm:.4f}")
    
    print(f"\n📊 Total demo samples: {len(demo_buffer)}")
    return demo_buffer