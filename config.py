# Environment settings

MATRIX_SIZE = 2                 # Size of the matrices to be multiplied (n x n)
VECTOR_LEN = MATRIX_SIZE ** 2   # 4
TENSOR_DIM = VECTOR_LEN ** 3    # 64

# Full exploration: allow up to 20 rank-1 tensors
MAX_STEPS = 10                  # Extended from 10 to allow full exploration

# --- Model Architecture ---
HIDDEN_DIM = 512           # Increased for better capacity (was 256)
N_HEADS = 3 * VECTOR_LEN    #  3 vectors * 4 entries = 12 heads
INPUT_DIM = TENSOR_DIM + 3  # 64 (residual tensor) + 3 (metadata: step, rank, norm)

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-4        # Lower LR for more stable exploration
BATCH_SIZE = 128            # Restored to 128 for stable learning
REPLAY_BUFFER_SIZE = 20000  # Larger buffer to store diverse solutions
EPOCHS = 200                # Restored to 200 for long-term training
EPISODES_PER_EPOCH = 10      # Restored to 10
MCTS_SIMS = 350             # Restored to 350 for deep thinking

# --- MCTS Settings ---
CPUCT = 2.0                 # Higher exploration constant
DIRICHLET_ALPHA = 0.6       # More uniform noise (encourages trying new actions)
DIRICHLET_EPSILON = 0.5     # 50% noise, 50% policy - high exploration


# --- Exploration Settings ---
EPSILON_GREEDY = 0.20       # 20% chance of random action during training
EFFICIENCY_BONUS = 5.0      # Strong bonus multiplier for finding fewer-step solutions
TARGET_RANK = 7             # Strassen's algorithm uses 7 multiplications

# --- Warm Start Settings ---
WARM_START_COPIES = 50      # Restored to 50
PRE_TRAIN_STEPS = 100       # Restored to 100

# --- Full Exploration Settings ---
EXPLORE_ALL_RANKS = True    # Enable exploration across all rank counts
MIN_RANK_TO_EXPLORE = 6     # Explore solutions from 6 steps
MAX_RANK_TO_EXPLORE = 20    # Up to 20 steps