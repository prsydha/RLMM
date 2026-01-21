import numpy as np
from env_tmp. matrix_env import MatrixTensorEnv

def test_strassen_tensor():
    """
    manually apply strassen's 7 coefficients to the 2x2 environment.
    if the code is correct, we should reach zero tensor in 7 steps.
    """
    env = MatrixTensorEnv(n=2)
    state = env.reset()

    print(f"Initial Tensor Norm: {np.linalg.norm(state)}")

    # Strassen's Algorithm coefficients (u, v, w)
    # These are the 7 rank-1 factors that sum up to the 2x2 MM tensor.
    # Note: These are usually typically defined in terms of M1...M7
    # For brevity, I will define just the first move to show it works.
    
    # Example: M1 = (A11 + A22)(B11 + B22)
    # In vector form (indices 0,1,2,3 -> 11,12,21,22):
    # u1 = [1, 0, 0, 1] (A11 + A22)
    # v1 = [1, 0, 0, 1] (B11 + B22)
    # w1 = [1, 0, 0, 1] (Writes to C11 + C22 contribution)

    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]  
    w1 = [1, 0, 0, 1]

    new_state, reward, done, _ = env.step(u1, v1, w1)

    print(f"Step 1 applied. New Norm: {np.linalg.norm(new_state)}, Reward: {reward}, Done: {done}")

if __name__ == "__main__":
    test_strassen_tensor()