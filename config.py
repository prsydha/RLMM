# Environment settings

MATRIX_SIZE = 2                 # Size of the matrices to be multiplied (n x n)
VECTOR_LEN = MATRIX_SIZE ** 2   # 9
TENSOR_DIM = VECTOR_LEN ** 3    # 729

# standard n^3 algorithm takes 27 steps for 3x3 matrix multiplication
MAX_STEPS = 10

# --- Model Architecture ---
HIDDEN_DIM = 256           # Number of neurons in hidden layers of the neural network ( increased from 256 to 512 for better capacity )
N_HEADS = 3 * VECTOR_LEN    #  3 vectors * 9 entries = 27 heads
INPUT_DIM = TENSOR_DIM      # 729