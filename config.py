# Environment settings

MATRIX_SIZE = 2  # Size of the matrices to be multiplied (n x n)
MAX_EPISODE_STEPS = MATRIX_SIZE**3 + 2  # Maximum steps per episode

# Agent Settings (for later)
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
GAMMA = 0.99     # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 500