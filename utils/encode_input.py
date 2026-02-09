import torch
import numpy as np

def encode_state(state):
    """
    Converts a tensor state with values [-1, 0, 1] into a one-hot representation.
    Input shape: (64,)
    Output shape: (192,)
    """
    # Flatten state if it isn't already
    flat_state = state.flatten()
    
    # Create an empty array for the one-hot version
    one_hot = np.zeros((len(flat_state), 3), dtype=np.float32)
    
    # Fill based on values: 
    # Index 0 for -1, Index 1 for 0, Index 2 for 1
    one_hot[flat_state == -1, 0] = 1.0
    one_hot[flat_state ==  0, 1] = 1.0
    one_hot[flat_state ==  1, 2] = 1.0
    
    return one_hot.flatten() # 64x3 = 192