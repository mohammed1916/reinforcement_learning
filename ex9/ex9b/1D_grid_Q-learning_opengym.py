import gym
from gym import spaces
import numpy as np

class OneDGridEnv(gym.Env):
    def __init__(self, grid_size=11, goal_state=10):
        super(OneDGridEnv, self).__init__()
        self.grid_size = grid_size
        self.goal_state = goal_state
        self.action_space = spaces.Discrete(2)  # Two actions: 0 = "left", 1 = "right"
        self.observation_space = spaces.Discrete(self.grid_size)  # 1D grid of grid_size cells
        self.state = 0  # Start at position 0

    def reset(self):
        self.state = 0  # Reset to start
        return self.state

    def step(self, action):
        if action == 0:  # Move left
            self.state = max(0, self.state - 1)
        elif action == 1:  # Move right
            self.state = min(self.grid_size - 1, self.state + 1)

        # Calculate reward
        reward = 10 if self.state == self.goal_state else -1

        # Check if the episode is done
        done = self.state == self.goal_state

        return self.state, reward, done, {}

    def render(self, reward):
        grid = ["_"] * self.grid_size
        grid[self.state] = "A"  # Mark the agent's current position
        grid[self.goal_state] = "G"  # Mark the goal position
        print(" ".join(grid), "Reward: ", reward)

# Instantiate the environment
env = OneDGridEnv()
# Initialize Q-table
num_states = env.observation_space.n
num_actions = env.action_space.n
Q = np.zeros((num_states, num_actions))

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.8  # Exploration rate

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # print("np.random.uniform()", np.random.uniform())
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state])  # Exploit

        # Take action and observe next state and reward
        next_state, reward, done, _ = env.step(action)

        # Update Q-table
        Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        state = next_state

# Evaluate the learned policy
num_eval_episodes = 10
for episode in range(num_eval_episodes):
    state = env.reset()
    done = False

    while not done:
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        env.render(reward)

env.close()

print(Q)
