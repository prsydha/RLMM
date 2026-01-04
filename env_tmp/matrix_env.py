import numpy as np

class MatrixTensorEnv:
    def __init__(self, n=2):
        self.n = n
        self.target_tensor = self._generate_mm_tensor(n)
        self.current_state = self.target_tensor.copy()
        self.steps = 0
        self.max_steps = n**3 + 2

    def _generate_mm_tensor(self, n):
        """
        Generates the tensor T for matrix multiplication C  = A * B. 
        T has shape( n*n, n*n, n*n)
        """
        dim = n * n
        T = np.zeros((dim, dim, dim))
        for i in range(n):
            for k in range(n):
                for j in range(n):
                    index_a = i * n + j # row major index for A
                    index_b = j * n + k # row major index for B
                    index_c = i * n + k # row major index for C

                    # set the entry in the tensor to 1
                    T[index_a, index_b, index_c] = 1
        return T
    
    def reset(self):
        self.current_state = self.target_tensor.copy()
        self.steps = 0
        return self.current_state
    
    def step(self, u, v, w):
        """
        Action: Apply a rank-1 update (u ⊗ v ⊗ w).
        u, v, w are vectors of size n^2.
        """

        # ensure u, v, w are numpy arrays
        u, v, w = np.array(u), np.array(v), np.array(w)
        
        # create rank-1 tensor from u, v, w
        # outer product : u ⊗ v ⊗ w
        update = np.einsum('i,j,k->ijk', u, v, w) # Einstein summation for outer product

        # update the state( residual tensor)
        self.current_state = self.current_state - update
        self.steps += 1

        # calculate reward
        # 1. exact match reward: if tensor is all zeros, massive reward.
        # 2. progress reward : negative of the norm of the residual tensor (encourage getting close to zero)
        residual_norm = np.linalg.norm(self.current_state)
        done = np.isclose(residual_norm, 0, atol=1e-5) # check if residual is close to zero upto certain absolute tolerance

        if done:
            reward = 100 - self.steps  # higher reward for fewer steps
        else:
            # penalize each step, and penalize remaining distance
            reward = -1 - residual_norm
        
        if self.steps >= self.max_steps:
            done = True
        
        return self.current_state, reward, done, {}
    
# --- verification block ---
# verify the tensor shape for 2x2 matrix multiplication
if __name__ == "__main__":
    env = MatrixTensorEnv(n=2)
    tensor = env.target_tensor
    print(f"Target Tensor Shape: {env.target_tensor.shape}")
    print(f"Total non-zero entries (should be 8 for standard 2x2): {np.sum(env.target_tensor)}")
    print(tensor)
                    
                     
