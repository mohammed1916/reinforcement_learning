import numpy as np

# Define the states
states = ["Distracted", "Study", "Take exam", "Obtain Certificate"]

# Define the transition probabilities matrix
P = np.array([
    [0.9, 0.1, 0.0, 0.0],  # Distracted -> [Distracted, Study, Take Exam, Obtain Certificate]
    [0.3, 0.5, 0.2, 0.0],  # Study -> [Distracted, Study, Take Exam, Obtain Certificate]
    [0.0, 0.4, 0.0, 0.6],  # Take Exam -> [Distracted, Study, Take Exam, Obtain Certificate]
    [0.0, 0.0, 0.0, 1.0]   # Obtain Certificate -> [Distracted, Study, Take Exam, Obtain Certificate]
])

# Define the reward matrix corresponding to each transition
R= np.array([
    [0, -3, 0, 0],    # Rewards for transitions from Distracted
    [-3, -2, -2, 0],    # Rewards for transitions from Study
    [0, 0, 0, 10],    # Rewards for transitions from Take Exam
    [0, 0, 0, 0]      # Rewards for transitions from Obtain Certificate
])
def simulate_markov(initial_state, P, R, steps=4):
    current_state = initial_state
    total_reward = 0
    state_history = [states[current_state]]

    for _ in range(steps):
        next_state = np.random.choice(len(states), p=P[current_state])
        reward = R[current_state, next_state]
        total_reward += reward
        state_history.append(states[next_state])
        current_state = next_state

        # If reached terminal state
        if current_state == len(states) - 1:
            break

    return state_history, total_reward

# Initial state: Distracted (index 0)
initial_state = 0
state_history, total_reward = simulate_markov(initial_state, P, R)

print("State history:", state_history)
print("Total reward:", total_reward)