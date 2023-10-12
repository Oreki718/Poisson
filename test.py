import numpy as np

def compute_divergence(R):
    # Initialize an array to store the divergence values
    div = np.zeros_like(R)

    # Compute the gradients P and Q using central differences
    P = np.zeros_like(R)
    Q = np.zeros_like(R)
    P[1:-1, :] = (R[2:, :] - R[:-2, :]) / 2
    Q[:, 1:-1] = (R[:, 2:] - R[:, :-2]) / 2

    # Handle the boundaries for P and Q using forward/backward differences
    P[0, :] = R[1, :] - R[0, :]
    P[-1, :] = R[-1, :] - R[-2, :]
    Q[:, 0] = R[:, 1] - R[:, 0]
    Q[:, -1] = R[:, -1] - R[:, -2]

    # Compute divergence using central differences
    div[1:-1, 1:-1] = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2 + (Q[1:-1, 2:] - Q[1:-1, :-2]) / 2

    # Handle the boundaries for div using forward/backward differences
    div[0, 1:-1] = P[1, 1:-1] - P[0, 1:-1] + (Q[0, 2:] - Q[0, :-2]) / 2
    div[-1, 1:-1] = P[-1, 1:-1] - P[-2, 1:-1] + (Q[-1, 2:] - Q[-1, :-2]) / 2
    div[1:-1, 0] = (P[2:, 0] - P[:-2, 0]) / 2 + Q[1:-1, 1] - Q[1:-1, 0]
    div[1:-1, -1] = (P[2:, -1] - P[:-2, -1]) / 2 + Q[1:-1, -1] - Q[1:-1, -2]

    # Handle the four corners using forward/backward differences
    div[0, 0] = P[1, 0] - P[0, 0] + Q[0, 1] - Q[0, 0]
    div[0, -1] = P[1, -1] - P[0, -1] + Q[0, -1] - Q[0, -2]
    div[-1, 0] = P[-1, 0] - P[-2, 0] + Q[-1, 1] - Q[-1, 0]
    div[-1, -1] = P[-1, -1] - P[-2, -1] + Q[-1, -1] - Q[-1, -2]

    return div

# Example
R = np.random.rand(4, 4)  # A sample 8x8 R channel
div_R = compute_divergence(R)
print(R)
print(div_R)
