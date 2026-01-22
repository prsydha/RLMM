import numpy as np


def create_target_tensor() -> np.ndarray:
        """
        Create the target tensor for matrix multiplication.

        For C = A × B where A is m×n and B is nxp:
        T[i,j,k] = 1 means: C[i,j] += A[i,k] * B[k,j]

        Returns:
            Target tensor of shape (mn, np, mp)
        """
        tensor = np.zeros((4, 4, 4), dtype=float)

        # Standard matrix multiplication tensor
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    idx_mn = i * 2 + j  # (i,j)
                    idx_np = j * 2 + k  # (j,k)
                    idx_mp = i * 2 + k  # (i,k)

                    tensor[idx_mn, idx_np, idx_mp] = 1.0

        return tensor

tensor = create_target_tensor()
a = np.array([0,0,0,-1])
b = np.array([0,1,-1,0])
c = np.array([-1,1,-1,1])
# Using einsum for n-way outer product
tensor_3d = np.einsum('i,j,k->ijk', a, b, c)
result = tensor - tensor_3d
print(result)

# print(tensor_3d)