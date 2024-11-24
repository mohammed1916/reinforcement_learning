import numpy as np
import random

# Define environment parameters
grid_size = 11  # grid from 0 to 10
goal_state = 10  # target cell
actions = ['left', 'right']
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.2  # exploration rate
num_episodes = 100  # number of training episodes

# Initialize Q-table with zeros
q_table = np.zeros((grid_size, len(actions)))

def get_next_state(state, action):
    # bound checks
    if action == 'left':
        return max(0, state - 1)
    elif action == 'right':
        return min(grid_size - 1, state + 1)

def get_reward(state):
    return 10 if state == goal_state else -1  # goal reward or small penalty

def choose_action(state):
    # Choose action with epsilon-greedy policy
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # exploration
    else:
        return actions[np.argmax(q_table[state])]  # exploitation

# Training loop
for episode in range(num_episodes):
    state = 0  # start at the beginning of the grid
    action = choose_action(state)
    done = False

    while not done:
        action_index = actions.index(action)
        
        # Take action, observe next state and reward
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)
        
        # Choose next action based on epsilon-greedy policy
        next_action = choose_action(next_state)
        next_action_index = actions.index(next_action)

        # Update Q-value using SARSA formula
        q_table[state, action_index] += alpha * (
            reward + gamma * q_table[next_state, next_action_index] - q_table[state, action_index]
        )

        # Transition to the next state and action
        state = next_state
        action = next_action

        # Check if the goal has been reached
        if state == goal_state:
            done = True

# Display the learned Q-table
print("Q-table after training:")
print(q_table)
