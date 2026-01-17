import numpy as np
a = np.array([1, 2])
b = np.array([3, 4, 5])
c = np.array([6, 7])
# Using einsum for n-way outer product
tensor_3d = np.einsum('i,j,k->ijk', a, b, c)

print(tensor_3d)