# Environment settings

MATRIX_SIZE = 3                # Size of the matrices to be multiplied (n x n)
VECTOR_LEN = MATRIX_SIZE ** 2   # 9
TENSOR_DIM = VECTOR_LEN ** 3    # 729

# Full exploration: allow up to 20 rank-1 tensors
MAX_STEPS = 30                  # Extended from 30 to allow full exploration

# --- Model Architecture ---
HIDDEN_DIM = 512           # Increased for better capacity (was 256)
N_HEADS = 3 * VECTOR_LEN    #  3 vectors * 9 entries = 27 heads
INPUT_DIM = TENSOR_DIM + 3  # 729 (residual tensor) + 3 (metadata: step, rank, norm)

# --- Training Hyperparameters ---
LEARNING_RATE = 1e-4        # Lower LR for more stable exploration
BATCH_SIZE = 128            # Larger batch for better gradient estimates
REPLAY_BUFFER_SIZE = 30000  # Larger buffer to store diverse solutions
EPOCHS = 300                # More epochs for exploration
EPISODES_PER_EPOCH = 10     # More episodes per epoch
MCTS_SIMS = 350             # Slightly more simulations for better search

# --- MCTS Settings ---
CPUCT = 2.0                 # Higher exploration constant
DIRICHLET_ALPHA = 0.8       # More uniform noise (encourages trying new actions)
DIRICHLET_EPSILON = 0.4     # 50% noise, 50% policy - high exploration


# --- Exploration Settings ---
EPSILON_GREEDY = 0.20       # 20% chance of random action during training
EFFICIENCY_BONUS = 5.0      # Strong bonus multiplier for finding fewer-step solutions
TARGET_RANK = 23            # Strassen's algorithm uses 7 multiplications

# --- Warm Start Settings ---
WARM_START_COPIES = 30      # More expert demo copies to learn from
PRE_TRAIN_STEPS = 100       # More pre-training to establish good baseline

# --- Full Exploration Settings ---
EXPLORE_ALL_RANKS = True    # Enable exploration across all rank counts
MIN_RANK_TO_EXPLORE = 22    # Explore solutions from 6 steps
MAX_RANK_TO_EXPLORE = 30    # Up to 20 steps