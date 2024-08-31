import numpy as np

# Simulate the Markov Reward Process
def simulate_mrp(P, num_episodes=10000, gamma=1.0):
    n = len(P)
    state_values = np.zeros(n)
    counts = np.zeros(n)
    
    for _ in range(num_episodes):
        state = 0  # Start from the first state (Distracted)
        total_reward = 0
        discount = 1.0
        while True:
            if state == n - 1:  # Terminal state (Obtain Certificate)
                break
            prob, reward = zip(*P[state])
            next_state = np.random.choice(range(n), p=prob)
            total_reward += discount * reward[next_state]
            discount *= gamma
            state = next_state
        
        state_values[state] += total_reward
        counts[state] += 1
    
    # Average the returns for each state
    for i in range(n):
        if counts[i] > 0:
            state_values[i] /= counts[i]
    
    return state_values

# Original P matrix containing tuples (probability, reward)
P = np.array([
    [(0.9, 0), (0.1, 0), (0,0),(0,0)],
    [(0.0, 0), (0.5, 0), (0.3, -3), (0.2, -2)],
    [(0.0, 0), (0.4, -1), (0.0, 0), (0.6, 10)],
    [(0.0, 0), (0.0, 0), (0.0, 0), (0.0, 0)]
])
# Simulate the MRP and compute state values
state_values = simulate_mrp(P)
print("Simulated State Values:", state_values)
