import numpy as np

# Original P matrix containing tuples (probability, reward)
P = np.array([
    [(0.9, 0), (0.1, 0), (0,0),(0,0)],
    [(0.0, 0), (0.5, 0), (0.3, -3), (0.2, -2)],
    [(0.0, 0), (0.4, -1), (0.0, 0), (0.6, 10)],
    [(0.0, 0), (0.0, 0), (0.0, 0), (0.0, 0)]
])
# P = np.array([(.9,0),(.1,0),(0,0),(0,0)],
#              [(.5,0),(.3,-3),(0.2,-2),(0,0)],
#              [(0,0),(0.4,-1),(0,0),(0.6,10)],
#              [(0,0),(0,0),(0,0),(0,0)],
#              )

# Extract the transition probability matrix P_prob and rewards R
n = len(P)
P_prob = np.zeros((n, n))
R = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        P_prob[i][j] = P[i][j][0]
        R[i][j] = P[i][j][1]

# Discount factor (gamma) set to 1 for simplicity
gamma = 1.0

# Compute the reward vector
reward_vector = np.sum(P_prob * R, axis=1)

# Identity matrix
I = np.eye(n)

# Compute the state values using the Bellman equation
V = np.linalg.solve(I - gamma * P_prob, reward_vector)

print("State Values:", V)
