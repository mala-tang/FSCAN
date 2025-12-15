import numpy as np

A = np.array([
    [0, 1, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 0]
], dtype=float)

X = np.array([
    [1.0, 2.0, 3.0],
    [0.5, 1.0, 1.5],
    [2.0, 1.0, 0.0],
    [1.5, 2.5, 3.5]
])

D = np.diag(np.sum(A, axis=1))
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(A, axis=1)))

I = np.eye(4)
print("I=\n", I)
L_sym = I - D_inv_sqrt @ A @ D_inv_sqrt
print("L_sym=\n", np.round(L_sym, 4))
Y = L_sym @ X

print("L_sym @ X =\n", np.round(Y, 4))