import numpy as np
import random

# Define environment parameters
grid_size = 11  # 0 to 10
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
    if state == goal_state:
        return 10  # reward for reaching the goal
    else:
        return -1  # small penalty for each move


def choose_action(state):
    # chooses a random action (exploration)
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)  # exploration
    # selects the action with the highest Q-value in the current state (exploitation).
    else:
        return actions[np.argmax(q_table[state])]  # exploitation


# Training loop
for episode in range(num_episodes):
    state = 0  # start at the beginning of the grid
    done = False

    while not done:
        # Choose action based on Q-table
        action = choose_action(state)
        action_index = actions.index(action)

        # Take action and observe next state and reward
        next_state = get_next_state(state, action)
        reward = get_reward(next_state)

        # Update Q-value using Q-learning formula
        best_next_action = np.argmax(q_table[next_state])
        # print("q_table",next_state, q_table[next_state])
        q_table[state, action_index] += alpha * (
                    reward + gamma * q_table[next_state, best_next_action] - q_table[state, action_index])

        # Transition to the next state
        state = next_state

        # Check if the goal has been reached
        if state == goal_state:
            done = True

# Display the learned Q-table
print("Q-table after training:")
print(q_table)