import numpy as np

# Define the transition probability matrix
P = np.array([
    [0.7, 0.3],
    [0.6, 0.4]])

# Initial state vector (Rainy on Monday)
initial_state = np.array([0, 1])

# Calculate the state vector after 1 transition (Tuesday)
state_after_1_day = np.dot(initial_state, P)

# Print the transition probability matrix
print("Transition Probability Matrix (TPM):")
print(P)

# Print the state vector after 1 day (Tuesday)
print("\nState vector on Tuesday given Rainy on Monday:")
print(state_after_1_day)

# Calculate the state vector after 2 transitions (Wednesday)
state_after_2_days = np.dot(initial_state, np.linalg.matrix_power(P,2))

# Probability of being rainy on Tuesday
prob_rainy_tuesday = state_after_1_day[1]
print("\nProbability of being rainy on Tuesday given Rainy on Monday:", prob_rainy_tuesday)

# Print the state vector after 2 days
print("\nState vector on Wednesday given Rainy on Monday:")
print(state_after_2_days)

# Probability of being rainy on Wednesday
prob_rainy_wednesday = state_after_2_days[1]
print("\nProbability of being rainy on Wednesday given Rainy on Monday:", prob_rainy_wednesday)